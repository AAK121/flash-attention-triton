"""FlashAttention-2 backward pass.

With S = sm_scale * Q K^T, P = softmax(S) and O = P V, the gradients are

    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - delta),  delta_i = sum_j P_ij dP_ij = rowsum(dO_i * O_i)
    dQ = sm_scale * dS K
    dK = sm_scale * dS^T Q

P is never stored: each kernel recomputes it from Q, K and the base-2
logsumexp saved by the forward pass, P = exp2(qk_scale * Q K^T - lse2).

The work is split into two kernels so that no output needs atomic updates:
one program per key block accumulates dK and dV over all query blocks, and one
program per query block accumulates dQ over all key blocks. The price is that
Q K^T and dO V^T are computed twice: 7 tile matmuls per (query, key) block pair
instead of 5. The alternative is a single pass over key blocks that adds each
partial dQ tile into an fp32 buffer in global memory with atomics. It avoids the
recomputation, but every key block then does a read-modify-write of its dQ rows,
contention grows with the number of key blocks, and the result depends on the
order in which the atomics land, so dQ is not bitwise reproducible. This
implementation chooses determinism and no atomic traffic.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._autotune import (
    KernelConfig,
    autotune_enabled,
    default_dkdv_config,
    default_dq_config,
    prune_dkdv,
    prune_dq,
    search_space,
)

NEG_INF = tl.constexpr(float("-inf"))
POS_INF = tl.constexpr(float("inf"))


@triton.jit
def _bwd_preprocess(
    o_ptr, do_ptr, delta_ptr,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh,
    H, Nq,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)
    row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = row_idx < Nq
    o = tl.load(
        o_ptr + off_b * stride_ob + off_h * stride_oh
        + row_idx[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=row_ok[:, None], other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + off_b * stride_dob + off_h * stride_doh
        + row_idx[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=row_ok[:, None], other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + off_b * stride_lb + off_h * stride_lh + row_idx, delta, mask=row_ok)


@triton.jit
def _dkdv_inner(
    dk, dv, k, v,
    q_base, do_base, lse_base, delta_base,
    stride_qm, stride_qd, stride_dom, stride_dod,
    qk_scale, col_idx, offs_d, Nq, causal_off,
    lo, hi,
    BLOCK_M: tl.constexpr,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Works in the transposed orientation (keys along rows) so that dK and dV
    # accumulate as (BLOCK_N, HEAD_DIM) tiles without a transpose per step.
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        row_idx = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_base + row_idx[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + row_idx[:, None] * stride_dom + offs_d[None, :] * stride_dod
        if MASKED:
            row_ok = row_idx < Nq
            q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=row_ok[:, None], other=0.0)
            lse = tl.load(lse_base + row_idx, mask=row_ok, other=POS_INF)
            delta = tl.load(delta_base + row_idx, mask=row_ok, other=0.0)
        else:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            lse = tl.load(lse_base + row_idx)
            delta = tl.load(delta_base + row_idx)
        # A query with no visible key has lse = -inf and P = 0 everywhere;
        # +inf makes exp2 return that 0 instead of inf.
        lse = tl.where(lse == NEG_INF, POS_INF, lse)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale
        if MASKED:
            keep = row_ok[None, :]
            if IS_CAUSAL:
                keep = keep & (col_idx[:, None] <= row_idx[None, :] + causal_off)
            # Masking the logits rather than P keeps exp2 from overflowing on
            # hidden positions whose raw score exceeds the row's lse.
            qkT = tl.where(keep, qkT, NEG_INF)
        pT = tl.math.exp2(qkT - lse[None, :])

        dv = tl.dot(pT.to(do.dtype), do, dv)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk)
    return dk, dv


@triton.jit
def _attn_bwd_dkdv(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh,
    H, Nq, Nk, sm_scale,
    seq_bucket,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Under a causal mask the first key blocks are seen by the most queries, so
    # the natural launch order already issues the heaviest programs first.
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    lse_base = lse_ptr + off_b * stride_lb + off_h * stride_lh
    delta_base = delta_ptr + off_b * stride_lb + off_h * stride_lh

    col_idx = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = col_idx < Nk
    k = tl.load(
        k_ptr + off_b * stride_kb + off_h * stride_kh
        + col_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=col_ok[:, None], other=0.0,
    )
    v = tl.load(
        v_ptr + off_b * stride_vb + off_h * stride_vh
        + col_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=col_ok[:, None], other=0.0,
    )
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Padded key rows (col >= Nk) only produce dK/dV rows that are never
    # stored, so the loop needs a mask only for query rows and the diagonal.
    causal_off = Nk - Nq
    end = tl.cdiv(Nq, BLOCK_M) * BLOCK_M
    full_rows_end = (Nq // BLOCK_M) * BLOCK_M
    first_col = start_n * BLOCK_N
    if IS_CAUSAL:
        # Query i sees key j when i >= j - causal_off. Queries before
        # first_col - causal_off see none of this block and are skipped; from
        # the first query that sees the last key of the block onward, every
        # key is visible and no elementwise mask is needed.
        lo = (tl.maximum(first_col - causal_off, 0) // BLOCK_M) * BLOCK_M
        lo = tl.minimum(lo, end)
        full_start = tl.cdiv(tl.maximum(first_col + BLOCK_N - 1 - causal_off, 0), BLOCK_M) * BLOCK_M
        full_start = tl.minimum(tl.maximum(full_start, lo), end)
    else:
        lo = 0
        full_start = 0

    # [lo, full_start): diagonal blocks. [full_start, full_rows_end): interior.
    # [max(full_start, full_rows_end), end): the ragged last query block.
    dk, dv = _dkdv_inner(
        dk, dv, k, v, q_base, do_base, lse_base, delta_base,
        stride_qm, stride_qd, stride_dom, stride_dod,
        qk_scale, col_idx, offs_d, Nq, causal_off,
        lo, full_start,
        BLOCK_M, True, IS_CAUSAL,
    )
    dk, dv = _dkdv_inner(
        dk, dv, k, v, q_base, do_base, lse_base, delta_base,
        stride_qm, stride_qd, stride_dom, stride_dod,
        qk_scale, col_idx, offs_d, Nq, causal_off,
        full_start, full_rows_end,
        BLOCK_M, False, IS_CAUSAL,
    )
    dk, dv = _dkdv_inner(
        dk, dv, k, v, q_base, do_base, lse_base, delta_base,
        stride_qm, stride_qd, stride_dom, stride_dod,
        qk_scale, col_idx, offs_d, Nq, causal_off,
        tl.maximum(full_start, full_rows_end), end,
        BLOCK_M, True, IS_CAUSAL,
    )

    dk = dk * sm_scale
    tl.store(
        dk_ptr + off_b * stride_dkb + off_h * stride_dkh
        + col_idx[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
        dk.to(dk_ptr.dtype.element_ty), mask=col_ok[:, None],
    )
    tl.store(
        dv_ptr + off_b * stride_dvb + off_h * stride_dvh
        + col_idx[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
        dv.to(dv_ptr.dtype.element_ty), mask=col_ok[:, None],
    )


@triton.jit
def _dq_inner(
    dq, q, do, lse, delta,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    qk_scale, row_idx, offs_d, Nk, causal_off,
    lo, hi,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + col_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + col_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            col_ok = col_idx < Nk
            k = tl.load(k_ptrs, mask=col_ok[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if MASKED:
            # Unlike dK/dV, padded keys here would add to real dQ rows.
            keep = col_ok[None, :]
            if IS_CAUSAL:
                keep = keep & (col_idx[None, :] <= row_idx[:, None] + causal_off)
            qk = tl.where(keep, qk, NEG_INF)
        p = tl.math.exp2(qk - lse[:, None])

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq)
    return dq


@triton.jit
def _attn_bwd_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lb, stride_lh,
    H, Nq, Nk, sm_scale,
    seq_bucket,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if IS_CAUSAL:
        # Same load balancing as the forward: late query blocks are heaviest.
        start_m = tl.num_programs(0) - 1 - tl.program_id(0)
    else:
        start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)

    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh

    row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = row_idx < Nq
    q = tl.load(
        q_ptr + off_b * stride_qb + off_h * stride_qh
        + row_idx[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=row_ok[:, None], other=0.0,
    )
    do = tl.load(
        do_ptr + off_b * stride_dob + off_h * stride_doh
        + row_idx[:, None] * stride_dom + offs_d[None, :] * stride_dod,
        mask=row_ok[:, None], other=0.0,
    )
    stats = off_b * stride_lb + off_h * stride_lh + row_idx
    lse = tl.load(lse_ptr + stats, mask=row_ok, other=POS_INF)
    lse = tl.where(lse == NEG_INF, POS_INF, lse)
    delta = tl.load(delta_ptr + stats, mask=row_ok, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Identical block bounds to the forward kernel.
    causal_off = Nk - Nq
    first_row = start_m * BLOCK_M
    if IS_CAUSAL:
        full_end = tl.maximum(tl.minimum(first_row + causal_off + 1, Nk), 0)
        hi = tl.maximum(tl.minimum(first_row + BLOCK_M + causal_off, Nk), 0)
    else:
        full_end = Nk
        hi = Nk
    full_hi = (full_end // BLOCK_N) * BLOCK_N

    dq = _dq_inner(
        dq, q, do, lse, delta, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        qk_scale, row_idx, offs_d, Nk, causal_off,
        0, full_hi,
        BLOCK_N, False, IS_CAUSAL,
    )
    dq = _dq_inner(
        dq, q, do, lse, delta, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        qk_scale, row_idx, offs_d, Nk, causal_off,
        full_hi, hi,
        BLOCK_N, True, IS_CAUSAL,
    )

    dq = dq * sm_scale
    tl.store(
        dq_ptr + off_b * stride_dqb + off_h * stride_dqh
        + row_idx[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
        dq.to(dq_ptr.dtype.element_ty), mask=row_ok[:, None],
    )


_attn_bwd_dkdv_tuned = triton.autotune(
    configs=search_space(),
    key=["HEAD_DIM", "IS_CAUSAL", "seq_bucket"],
    prune_configs_by={"early_config_prune": prune_dkdv},
)(_attn_bwd_dkdv)

_attn_bwd_dq_tuned = triton.autotune(
    configs=search_space(),
    key=["HEAD_DIM", "IS_CAUSAL", "seq_bucket"],
    prune_configs_by={"early_config_prune": prune_dq},
)(_attn_bwd_dq)

# delta is a bandwidth-bound row reduction; tuning it would not pay off.
_PREPROCESS_BLOCK_M = 64


def _launch(kernel, tuned, config, default, grid_fn, args, head_dim, causal):
    if config is None and autotune_enabled():
        tuned[lambda meta: grid_fn(meta["BLOCK_M"], meta["BLOCK_N"])](
            *args, HEAD_DIM=head_dim, IS_CAUSAL=causal
        )
        return
    cfg = config or default(head_dim)
    kernel[grid_fn(cfg.block_m, cfg.block_n)](
        *args,
        HEAD_DIM=head_dim, IS_CAUSAL=causal,
        BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )


def attn_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
    sm_scale: float | None,
    dkdv_config: KernelConfig | None = None,
    dq_config: KernelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) given dO, the forward inputs, O and the base-2 lse."""
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    if Nq == 0 or Nk == 0 or B * H == 0:
        return dq.zero_(), dk.zero_(), dv.zero_()

    delta = torch.empty_like(lse)
    _bwd_preprocess[(triton.cdiv(Nq, _PREPROCESS_BLOCK_M), B * H)](
        o, do, delta,
        *o.stride(), *do.stride(),
        delta.stride(0), delta.stride(1),
        H, Nq,
        HEAD_DIM=D, BLOCK_M=_PREPROCESS_BLOCK_M,
    )

    seq_bucket = triton.next_power_of_2(max(Nq, Nk))
    common = (*q.stride(), *k.stride(), *v.stride(), *do.stride())
    _launch(
        _attn_bwd_dkdv, _attn_bwd_dkdv_tuned, dkdv_config, default_dkdv_config,
        lambda bm, bn: (triton.cdiv(Nk, bn), B * H),
        (
            q, k, v, do, lse, delta, dk, dv,
            *common, *dk.stride(), *dv.stride(),
            lse.stride(0), lse.stride(1),
            H, Nq, Nk, sm_scale, seq_bucket,
        ),
        D, causal,
    )
    _launch(
        _attn_bwd_dq, _attn_bwd_dq_tuned, dq_config, default_dq_config,
        lambda bm, bn: (triton.cdiv(Nq, bm), B * H),
        (
            q, k, v, do, lse, delta, dq,
            *common, *dq.stride(),
            lse.stride(0), lse.stride(1),
            H, Nq, Nk, sm_scale, seq_bucket,
        ),
        D, causal,
    )
    return dq, dk, dv
