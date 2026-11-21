from __future__ import annotations

import math

import pytest
import torch

from flash_attn_triton import attention_reference, flash_attention
from flash_attn_triton._autotune import KernelConfig
from flash_attn_triton._bwd import attn_backward
from flash_attn_triton._fwd import attn_forward

from .helpers import (
    assert_close_to_reference,
    assert_grads_close,
    kernel_grads,
    low_precision_reference,
)

F16 = torch.float16


def make_gqa(b, hq, hk, nq, nk, d, device, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=F16)

    return rnd(b, hq, nq, d), rnd(b, hk, nk, d), rnd(b, hk, nk, d), rnd(b, hq, nq, d)


def check(q, k, v, do, causal, window, what):
    out, grads = kernel_grads(flash_attention, q, k, v, do, causal=causal, window_size=window)
    ref = attention_reference(q, k, v, causal=causal, window_size=window)
    low = attention_reference(q, k, v, causal=causal, window_size=window, compute_dtype=F16)
    assert_close_to_reference(out, ref, low, F16, f"{what} out")
    assert_grads_close(q, k, v, do, causal, grads, F16, what, window=window)


@pytest.mark.parametrize("hq,hk", [(4, 2), (6, 1), (8, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nq,nk", [(17, 17), (200, 200), (20, 150), (150, 20), (1000, 1000)])
def test_grouped_query(hq, hk, nq, nk, causal, device, require_size):
    # dK and dV of a shared head must sum the contributions of every query
    # head in its group; the reference expands K and V with repeat_interleave,
    # so autograd performs that sum for it.
    require_size(nq, nk)
    q, k, v, do = make_gqa(2, hq, hk, nq, nk, 64, device, seed=30)
    check(q, k, v, do, causal, None, f"gqa {hq}/{hk} nq={nq} nk={nk}")


def test_grouped_query_matches_expanded_heads(device, require_size):
    # GQA must equal ordinary attention on explicitly repeated K and V, with
    # dK/dV equal to the group sum of the expanded gradients.
    require_size(64)
    q, k, v, do = make_gqa(1, 6, 2, 64, 64, 32, device, seed=31)
    out, (dq, dk, dv) = kernel_grads(flash_attention, q, k, v, do, causal=True)
    ke, ve = k.repeat_interleave(3, dim=1), v.repeat_interleave(3, dim=1)
    out_e, (dq_e, dk_e, dv_e) = kernel_grads(flash_attention, q, ke, ve, do, causal=True)
    assert torch.equal(out, out_e)
    assert torch.equal(dq, dq_e)
    sum_k = dk_e.float().view(1, 2, 3, 64, 32).sum(2)
    sum_v = dv_e.float().view(1, 2, 3, 64, 32).sum(2)
    # The kernel sums in fp32 before rounding once; the expanded path rounds
    # each head's gradient to fp16 first, so allow a few rounding steps.
    tol = 4 * 2.0**-11 * max(sum_k.abs().max().item(), sum_v.abs().max().item())
    assert (dk.float() - sum_k).abs().max().item() <= tol
    assert (dv.float() - sum_v).abs().max().item() <= tol


WINDOWS = [(0, 0), (8, 8), (5, 0), (33, -1), (-1, 16), (64, 3)]
WINDOW_SHAPES = [(1, 1), (37, 37), (200, 200), (20, 150), (150, 20), (1023, 1023)]


@pytest.mark.parametrize("window", WINDOWS, ids=str)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nq,nk", WINDOW_SHAPES)
def test_sliding_window(window, nq, nk, causal, device, require_size):
    require_size(nq, nk)
    q, k, v, do = make_gqa(1, 2, 2, nq, nk, 64, device, seed=32)
    check(q, k, v, do, causal, window, f"window={window} causal={causal} nq={nq} nk={nk}")


def test_window_wider_than_sequence_is_full_attention(device, require_size):
    require_size(100)
    q, k, v, _ = make_gqa(1, 2, 2, 100, 100, 32, device, seed=33)
    a = flash_attention(q, k, v, window_size=(500, 500))
    b = flash_attention(q, k, v)
    ref = attention_reference(q, k, v)
    low = low_precision_reference(q, k, v, False)
    assert_close_to_reference(a, ref, low, F16, "wide window")
    assert_close_to_reference(b, ref, low, F16, "full")


def test_causal_window_equals_causal_with_left_limit(device, require_size):
    # causal=True with window (left, anything) is the window (left, 0).
    require_size(80)
    q, k, v, _ = make_gqa(1, 2, 2, 80, 80, 32, device, seed=34)
    a = flash_attention(q, k, v, causal=True, window_size=(10, 7))
    b = flash_attention(q, k, v, window_size=(10, 0))
    assert torch.equal(a, b)


BLOCK_CONFIGS = [
    KernelConfig(16, 16, num_warps=1),
    KernelConfig(16, 64, num_warps=1),
    KernelConfig(64, 16, num_warps=1),
    KernelConfig(32, 128, num_warps=2),
    KernelConfig(128, 32, num_warps=2),
]


@pytest.mark.parametrize("cfg", BLOCK_CONFIGS, ids=lambda c: f"{c.block_m}x{c.block_n}")
@pytest.mark.parametrize("window", [(3, 7), (17, 0), (40, 40), (0, 31)], ids=str)
@pytest.mark.parametrize("nq,nk", [(37, 37), (130, 130), (31, 48), (130, 128), (20, 150)])
def test_window_block_configs(cfg, window, nq, nk, device, require_size):
    # Windows whose edges fall inside, on and across tile boundaries, for tile
    # shapes where the window spans one or several tiles.
    require_size(nq, nk)
    q, k, v, do = make_gqa(1, 4, 2, nq, nk, 32, device, seed=35)
    o, lse = attn_forward(q, k, v, False, None, window, config=cfg)
    ref, lse_ref = attention_reference(q, k, v, window_size=window, return_lse=True)
    low = attention_reference(q, k, v, window_size=window, compute_dtype=F16)
    assert_close_to_reference(o, ref, low, F16, f"{cfg} {window} out")
    visible = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), visible)
    assert (lse[visible] * math.log(2.0) - lse_ref[visible]).abs().max().item() < 2e-3
    grads = attn_backward(
        do, q, k, v, o, lse, False, None, window, dkdv_config=cfg, dq_config=cfg
    )
    assert_grads_close(q, k, v, do, False, grads, F16, f"{cfg} {window}", window=window)


def test_rejects_bad_head_counts_and_windows(device):
    q, k, v, _ = make_gqa(1, 4, 3, 8, 8, 32, device, seed=36)
    with pytest.raises(ValueError):
        flash_attention(q, k, v)
    q, k, v, _ = make_gqa(1, 4, 2, 8, 8, 32, device, seed=36)
    with pytest.raises(ValueError):
        flash_attention(q, k, v, window_size=(-2, 0))
