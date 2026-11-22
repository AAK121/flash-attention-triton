from __future__ import annotations

import pytest
import torch

from flash_attn_triton import flash_attention, flash_attention_varlen

from .helpers import assert_close_to_reference, reference_grads

F16 = torch.float16


def packed(lens_q, lens_k, hq, hk, d, device, seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=F16)

    def offsets(lens):
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(lens).cumsum(0)
        return cu.to(device)

    q = rnd(sum(lens_q), hq, d)
    k, v = rnd(sum(lens_k), hk, d), rnd(sum(lens_k), hk, d)
    do = rnd(sum(lens_q), hq, d)
    return q, k, v, do, offsets(lens_q), offsets(lens_k)


def per_sequence(t, cu, b):
    # Packed (tokens, heads, dim) slice -> dense (1, heads, seq, dim) view.
    return t[cu[b] : cu[b + 1]].transpose(0, 1).unsqueeze(0)


CASES = [
    ([5, 70, 1, 33], [9, 64, 40, 33]),
    ([17, 17, 17], [17, 17, 17]),
    ([1, 200, 64], [130, 3, 64]),
]


@pytest.mark.parametrize("lens_q,lens_k", CASES)
@pytest.mark.parametrize("causal,window", [(False, None), (True, None), (False, (7, 2))])
@pytest.mark.parametrize("hq,hk", [(4, 4), (4, 2)])
def test_varlen_matches_each_sequence(lens_q, lens_k, causal, window, hq, hk, device, require_size):
    require_size(max(lens_q), max(lens_k))
    q, k, v, do, cu_q, cu_k = packed(lens_q, lens_k, hq, hk, 32, device, seed=40)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    o = flash_attention_varlen(
        *leaves, cu_q, cu_k, max(lens_q), max(lens_k), causal=causal, window_size=window
    )
    o.backward(do)
    for b in range(len(lens_q)):
        qs, ks, vs, dos = (per_sequence(t, cu, b) for t, cu in
                           ((q, cu_q), (k, cu_k), (v, cu_k), (do, cu_q)))
        ref_o, ref_g = reference_grads(qs, ks, vs, dos, causal, window=window)
        low_o, low_g = reference_grads(qs, ks, vs, dos, causal, compute_dtype=F16, window=window)
        what = f"seq {b} ({lens_q[b]}x{lens_k[b]})"
        assert_close_to_reference(per_sequence(o.detach(), cu_q, b), ref_o, low_o, F16, what)
        scale = max(r.abs().max().item() for r in ref_g)
        got = [per_sequence(leaves[0].grad, cu_q, b), per_sequence(leaves[1].grad, cu_k, b),
               per_sequence(leaves[2].grad, cu_k, b)]
        for name, g, r, lo in zip(("dq", "dk", "dv"), got, ref_g, low_g):
            assert_close_to_reference(g, r, lo, F16, f"{what} {name}", scale=scale)


def test_varlen_equal_lengths_matches_dense(device, require_size):
    # With every sequence the same length, packing is only a layout change.
    require_size(48)
    b, n, h, d = 3, 48, 2, 64
    q, k, v, do, cu_q, cu_k = packed([n] * b, [n] * b, h, h, d, device, seed=41)
    out = flash_attention_varlen(q, k, v, cu_q, cu_k, n, n, causal=True)
    dense = flash_attention(
        *(t.view(b, n, h, d).transpose(1, 2) for t in (q, k, v)), causal=True
    )
    assert torch.equal(out.view(b, n, h, d).transpose(1, 2), dense)


def test_varlen_rejects_bad_offsets(device):
    q, k, v, _, cu_q, cu_k = packed([4, 4], [4, 4], 2, 2, 32, device, seed=42)
    with pytest.raises(ValueError):
        flash_attention_varlen(q, k, v, cu_q.long(), cu_k, 4, 4)
    with pytest.raises(ValueError):
        flash_attention_varlen(q, k, v, cu_q[:2], cu_k, 4, 4)
