from __future__ import annotations

import pytest
import torch

from flash_attn_triton import flash_attention
from flash_attn_triton._autotune import KernelConfig
from flash_attn_triton._bwd import attn_backward
from flash_attn_triton._fwd import attn_forward

from .conftest import HAS_CUDA, INTERPRET
from .helpers import assert_grads_close, kernel_grads, make_qkv


def _do(b, h, n, d, dtype, device, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, n, d, generator=g).to(device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("b,h", [(1, 1), (2, 3)])
@pytest.mark.parametrize("n", [1, 17, 200, 1000, 1023, 1024, 4096])
def test_gradients_match_reference(n, b, h, d, causal, dtype, device, require_dtype, require_size):
    require_dtype(dtype)
    require_size(n)
    if n >= 4096 and (b, h) != (1, 1):
        pytest.skip("covered by the single-head case")
    q, k, v = make_qkv(b, h, n, n, d, dtype, device, seed=10)
    do = _do(b, h, n, d, dtype, device, seed=11)
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=causal)
    assert_grads_close(q, k, v, do, causal, grads, dtype, f"n={n} d={d} causal={causal}")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "nq,nk",
    [(1, 200), (17, 1000), (200, 17), (64, 1024), (1000, 1023), (5, 3), (40, 8)],
)
def test_gradients_unequal_lengths(nq, nk, causal, device, require_size):
    # (200, 17), (5, 3) and (40, 8) under causal include queries with no
    # visible key, whose gradients must be exactly zero rather than NaN.
    require_size(nq, nk)
    q, k, v = make_qkv(2, 2, nq, nk, 64, torch.float16, device, seed=12)
    do = _do(2, 2, nq, 64, torch.float16, device, seed=13)
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=causal)
    assert_grads_close(q, k, v, do, causal, grads, torch.float16, f"nq={nq} nk={nk}")
    if causal and nq > nk:
        assert (grads[0][:, :, : nq - nk].float() == 0).all()


BLOCK_CONFIGS = [
    KernelConfig(16, 16, num_warps=1),
    KernelConfig(16, 64, num_warps=4),
    KernelConfig(64, 16, num_warps=1),
    KernelConfig(32, 128, num_warps=8),
    KernelConfig(128, 32, num_warps=2),
]
# Same tile-boundary offsets as the forward block test.
BLOCK_SHAPES = [(1, 1), (37, 37), (130, 130), (20, 150), (150, 20), (31, 48), (64, 129), (130, 128)]


@pytest.mark.parametrize("cfg", BLOCK_CONFIGS, ids=lambda c: f"{c.block_m}x{c.block_n}")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nq,nk", BLOCK_SHAPES)
def test_backward_block_configs(cfg, nq, nk, causal, device, require_size):
    # The same config drives both kernels; it is a key tile of BLOCK_N rows
    # for dK/dV and a query tile of BLOCK_M rows for dQ, so the warp count
    # has to suit the smaller of the two.
    require_size(nq, nk)
    q, k, v = make_qkv(1, 2, nq, nk, 32, torch.float16, device, seed=14)
    do = _do(1, 2, nq, 32, torch.float16, device, seed=15)
    o, lse = attn_forward(q, k, v, causal, None)
    grads = attn_backward(do, q, k, v, o, lse, causal, None, dkdv_config=cfg, dq_config=cfg)
    assert_grads_close(q, k, v, do, causal, grads, torch.float16, f"{cfg} nq={nq} nk={nk}")


@pytest.mark.parametrize("causal", [False, True])
def test_gradients_non_contiguous(causal, device, require_size):
    require_size(50)
    b, h, n, d = 2, 3, 50, 64
    g = torch.Generator().manual_seed(16)

    def bshd(*shape):
        return torch.randn(*shape, generator=g).half().to(device).transpose(1, 2)

    q, k, v = bshd(b, n, h, d), bshd(b, n, h, d), bshd(b, n, h, d)
    # dO arrives in whatever layout the next op produced; here a transposed view.
    do = bshd(b, n, h, d)
    assert not do.is_contiguous()
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=causal)
    assert_grads_close(q, k, v, do, causal, grads, torch.float16, "non-contiguous")


def test_gradients_custom_sm_scale(device, require_size):
    require_size(33)
    q, k, v = make_qkv(1, 2, 33, 33, 32, torch.float16, device, seed=17)
    do = _do(1, 2, 33, 32, torch.float16, device, seed=18)
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=True, sm_scale=0.3)
    assert_grads_close(q, k, v, do, True, grads, torch.float16, "sm_scale", sm_scale=0.3)


@pytest.mark.parametrize("causal", [False, True])
def test_gradients_large_logits(causal, device, require_size):
    require_size(64)
    q, k, v = make_qkv(1, 2, 64, 64, 64, torch.float16, device, seed=19, scale=8.0)
    do = _do(1, 2, 64, 64, torch.float16, device, seed=20)
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=causal)
    assert_grads_close(q, k, v, do, causal, grads, torch.float16, "large logits")


def test_backward_deterministic(device, require_size):
    # No atomics anywhere, so repeated runs must agree bit for bit.
    require_size(130)
    q, k, v = make_qkv(2, 2, 130, 130, 64, torch.float16, device, seed=21)
    do = _do(2, 2, 130, 64, torch.float16, device, seed=22)
    _, first = kernel_grads(flash_attention, q, k, v, do, causal=True)
    for _ in range(3):
        _, again = kernel_grads(flash_attention, q, k, v, do, causal=True)
        for a, b in zip(first, again):
            assert torch.equal(a, b)


def test_partial_requires_grad(device):
    q, k, v = make_qkv(1, 2, 20, 20, 32, torch.float16, device, seed=23)
    q.requires_grad_()
    flash_attention(q, k, v, causal=True).sum().backward()
    assert q.grad is not None and k.grad is None and v.grad is None


@pytest.mark.skipif(not HAS_CUDA or INTERPRET, reason="autotuning needs a GPU")
def test_autotuned_backward(device, monkeypatch):
    monkeypatch.setenv("FLASH_ATTN_TRITON_AUTOTUNE", "1")
    q, k, v = make_qkv(2, 4, 1000, 1000, 128, torch.float16, device, seed=24)
    do = _do(2, 4, 1000, 128, torch.float16, device, seed=25)
    _, grads = kernel_grads(flash_attention, q, k, v, do, causal=True)
    assert_grads_close(q, k, v, do, True, grads, torch.float16, "autotuned")
