from __future__ import annotations

import math

import pytest
import torch

from flash_attn_triton import attention_reference, flash_attention
from flash_attn_triton._autotune import KernelConfig
from flash_attn_triton._fwd import attn_forward

from .conftest import HAS_CUDA, INTERPRET
from .helpers import assert_close_to_reference, low_precision_reference, make_qkv

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("b,h", [(1, 1), (2, 3)])
@pytest.mark.parametrize("n", [1, 17, 200, 1000, 1023, 1024, 4096])
def test_forward_matches_reference(n, b, h, d, causal, dtype, device, require_dtype, require_size):
    require_dtype(dtype)
    require_size(n)
    if n >= 4096 and (b, h) != (1, 1):
        pytest.skip("covered by the single-head case")
    q, k, v = make_qkv(b, h, n, n, d, dtype, device)
    out = flash_attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    low = low_precision_reference(q, k, v, causal)
    assert_close_to_reference(out, ref, low, dtype, f"n={n} d={d} causal={causal}")


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "nq,nk",
    [(1, 200), (17, 1000), (200, 17), (64, 1024), (1000, 1023), (5, 3)],
)
def test_unequal_lengths(nq, nk, causal, dtype, device, require_size):
    require_size(nq, nk)
    q, k, v = make_qkv(2, 2, nq, nk, 64, dtype, device, seed=1)
    out = flash_attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    low = low_precision_reference(q, k, v, causal)
    assert torch.isfinite(out.float()).all()
    assert_close_to_reference(out, ref, low, dtype, f"nq={nq} nk={nk} causal={causal}")


@pytest.mark.parametrize("causal", [False, True])
def test_logsumexp(causal, device, require_size):
    require_size(200)
    q, k, v = make_qkv(1, 2, 200, 200, 64, torch.float16, device, seed=2)
    _, lse2 = attn_forward(q, k, v, causal, None)
    _, lse_ref = attention_reference(q, k, v, causal=causal, return_lse=True)
    lse = lse2 * math.log(2.0)
    assert (lse - lse_ref).abs().max().item() < 2e-3


def test_rows_without_visible_keys_are_zero(device, require_size):
    # nq > nk with bottom-right alignment: the first nq - nk rows see nothing.
    require_size(40)
    q, k, v = make_qkv(1, 1, 40, 8, 32, torch.float16, device, seed=3)
    out = flash_attention(q, k, v, causal=True)
    assert torch.isfinite(out.float()).all()
    assert (out[:, :, :32].float() == 0).all()
    ref = attention_reference(q, k, v, causal=True)
    assert (out.float() - ref).abs().max().item() < 5e-3


@pytest.mark.parametrize("causal", [False, True])
def test_non_contiguous_inputs(causal, device, require_size):
    require_size(50)
    b, h, n, d = 2, 3, 50, 64
    g = torch.Generator().manual_seed(4)
    # Q and K: (batch, seq, heads, dim) storage viewed as (batch, heads, seq, dim),
    # the layout a fused QKV projection produces before any .contiguous().
    q = torch.randn(b, n, h, d, generator=g).half().to(device).transpose(1, 2)
    k = torch.randn(b, n, h, d, generator=g).half().to(device).transpose(1, 2)
    # V: every other column of a wider tensor, so the head-dim stride is 2.
    v = torch.randn(b, h, n, 2 * d, generator=g).half().to(device)[..., ::2]
    assert not q.is_contiguous() and not k.is_contiguous() and v.stride(-1) == 2
    out = flash_attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    low = low_precision_reference(q, k, v, causal)
    assert_close_to_reference(out, ref, low, torch.float16, "non-contiguous")


# Unequal BLOCK_M and BLOCK_N are where the causal block bounds are easiest to
# get wrong: the diagonal then crosses several key tiles of one query tile, or
# one key tile spans several query tiles.
BLOCK_CONFIGS = [
    KernelConfig(16, 16, num_warps=1),
    KernelConfig(16, 64, num_warps=1),
    KernelConfig(64, 16, num_warps=4),
    KernelConfig(32, 128, num_warps=2),
    KernelConfig(128, 32, num_warps=8),
    KernelConfig(64, 64, num_warps=4),
]


# Offsets nk - nq of 1 and -2 (mod 16) put the causal diagonal exactly on a tile
# boundary, which is what an off-by-one in the loop bounds needs to show up.
BLOCK_SHAPES = [
    (1, 1), (37, 37), (130, 130), (20, 150), (150, 20), (127, 129),
    (31, 48), (64, 129), (130, 128), (40, 54),
]


@pytest.mark.parametrize("cfg", BLOCK_CONFIGS, ids=lambda c: f"{c.block_m}x{c.block_n}")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nq,nk", BLOCK_SHAPES)
def test_block_configs(cfg, nq, nk, causal, device, require_size):
    require_size(nq, nk)
    q, k, v = make_qkv(1, 2, nq, nk, 32, torch.float16, device, seed=8)
    out, lse2 = attn_forward(q, k, v, causal, None, config=cfg)
    ref, lse_ref = attention_reference(q, k, v, causal=causal, return_lse=True)
    low = low_precision_reference(q, k, v, causal)
    assert_close_to_reference(out, ref, low, torch.float16, f"{cfg} nq={nq} nk={nk}")
    visible = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse2), visible)
    assert (lse2[visible] * math.log(2.0) - lse_ref[visible]).abs().max().item() < 2e-3


@pytest.mark.skipif(not HAS_CUDA or INTERPRET, reason="autotuning needs a GPU")
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_autotuned_path(d, causal, device, monkeypatch):
    monkeypatch.setenv("FLASH_ATTN_TRITON_AUTOTUNE", "1")
    q, k, v = make_qkv(2, 4, 1000, 1000, d, torch.float16, device, seed=9)
    out = flash_attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    low = low_precision_reference(q, k, v, causal)
    assert_close_to_reference(out, ref, low, torch.float16, f"autotuned d={d}")


def test_empty_sequences(device):
    q, k, v = make_qkv(1, 2, 0, 5, 32, torch.float16, device)
    assert flash_attention(q, k, v).shape == (1, 2, 0, 32)
    q, k, v = make_qkv(1, 2, 5, 0, 32, torch.float16, device)
    out, lse2 = attn_forward(q, k, v, False, None)
    assert (out == 0).all() and torch.isinf(lse2).all()


def test_custom_sm_scale(device, require_size):
    require_size(33)
    q, k, v = make_qkv(1, 2, 33, 33, 32, torch.float16, device, seed=5)
    out = flash_attention(q, k, v, causal=True, sm_scale=0.3)
    ref = attention_reference(q, k, v, causal=True, sm_scale=0.3)
    low = low_precision_reference(q, k, v, True, sm_scale=0.3)
    assert_close_to_reference(out, ref, low, torch.float16, "sm_scale")


@pytest.mark.parametrize("causal", [False, True])
def test_large_logits_are_finite(causal, device, require_size):
    require_size(64)
    # Logits reach the hundreds, where exp without max subtraction overflows fp16.
    q, k, v = make_qkv(1, 2, 64, 64, 64, torch.float16, device, seed=6, scale=8.0)
    out = flash_attention(q, k, v, causal=causal)
    ref = attention_reference(q, k, v, causal=causal)
    assert torch.isfinite(out.float()).all()
    low = low_precision_reference(q, k, v, causal)
    assert_close_to_reference(out, ref, low, torch.float16, "large logits")


def test_deterministic(device, require_size):
    require_size(64)
    q, k, v = make_qkv(2, 2, 64, 64, 64, torch.float16, device, seed=7)
    a = flash_attention(q, k, v, causal=True)
    for _ in range(3):
        assert torch.equal(a, flash_attention(q, k, v, causal=True))


def test_input_validation(device):
    q, k, v = make_qkv(1, 1, 8, 8, 32, torch.float16, device)
    with pytest.raises(ValueError):
        flash_attention(q.float(), k.float(), v.float())
    with pytest.raises(ValueError):
        flash_attention(q[..., :24], k[..., :24], v[..., :24])
    with pytest.raises(ValueError):
        flash_attention(q, k, v[:, :, :4])


@pytest.mark.skipif(INTERPRET, reason="CPU tensors are valid in interpreter mode")
def test_rejects_cpu_tensors():
    q, k, v = make_qkv(1, 1, 8, 8, 32, torch.float16, torch.device("cpu"))
    with pytest.raises(ValueError):
        flash_attention(q, k, v)
