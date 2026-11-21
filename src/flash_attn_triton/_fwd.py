"""FlashAttention-2 forward kernel.

One program handles one (batch, head, query block). It streams over key/value
blocks, keeping a running row maximum m_i, a running row sum l_i and an
unnormalized output accumulator in registers, so the N x N score matrix is never
written to memory.

All logits are kept in the base-2 domain: the softmax scale is multiplied by
log2(e) once, and exp2 is used throughout because it maps to a single hardware
instruction. The logsumexp saved for the backward pass is therefore also in
base 2.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._autotune import (
    KernelConfig,
    autotune_enabled,
    default_forward_config,
    forward_configs,
    prune_forward,
)

# Kernels may only read module globals that are declared constexpr.
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    qk_scale, row_idx, offs_d, Nk, causal_off,
    lo, hi,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # MASKED=False is only used for blocks that are entirely inside the valid
    # key range and entirely below the causal diagonal, so no per-element
    # predicate is needed on the loads or on the scores.
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        col_idx = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_d[:, None] * stride_kd + col_idx[None, :] * stride_kn
        v_ptrs = v_base + col_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASKED:
            col_ok = col_idx < Nk
            k = tl.load(k_ptrs, mask=col_ok[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=col_ok[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

        s = tl.dot(q, k) * qk_scale
        if MASKED:
            keep = col_ok[None, :]
            if IS_CAUSAL:
                keep = keep & (col_idx[None, :] <= row_idx[:, None] + causal_off)
            s = tl.where(keep, s, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A row with no visible key so far has m_new = -inf; subtracting it would
        # give inf - inf. Substituting 0 makes every exp2 below evaluate to 0.
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh,
    H, Nq, Nk, sm_scale,
    seq_bucket,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if IS_CAUSAL:
        # Under a causal mask the last query blocks do the most work. Issuing
        # them first lets the short blocks fill in the tail of the launch.
        start_m = tl.num_programs(0) - 1 - tl.program_id(0)
    else:
        start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    lse_base = lse_ptr + off_b * stride_lb + off_h * stride_lh

    row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = row_idx < Nq

    q = tl.load(
        q_base + row_idx[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=row_ok[:, None], other=0.0,
    )

    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Query i sees keys j <= i + causal_off (bottom-right alignment).
    causal_off = Nk - Nq
    first_row = start_m * BLOCK_M
    if IS_CAUSAL:
        # Keys up to first_row + causal_off are visible to every row of this
        # block, so whole BLOCK_N tiles inside that range need no mask.
        full_end = tl.minimum(first_row + causal_off + 1, Nk)
        full_end = tl.maximum(full_end, 0)
        hi = tl.minimum(first_row + BLOCK_M + causal_off, Nk)
        hi = tl.maximum(hi, 0)
    else:
        full_end = Nk
        hi = Nk
    full_hi = (full_end // BLOCK_N) * BLOCK_N

    acc, l_i, m_i = _fwd_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        qk_scale, row_idx, offs_d, Nk, causal_off,
        0, full_hi,
        BLOCK_N, False, IS_CAUSAL,
    )
    acc, l_i, m_i = _fwd_inner(
        acc, l_i, m_i, q, k_base, v_base,
        stride_kn, stride_kd, stride_vn, stride_vd,
        qk_scale, row_idx, offs_d, Nk, causal_off,
        full_hi, hi,
        BLOCK_N, True, IS_CAUSAL,
    )

    # Rows that saw no key at all keep l_i = 0; emit zeros and lse = -inf.
    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = tl.where(empty, NEG_INF, m_i + tl.math.log2(l_safe))

    tl.store(
        o_base + row_idx[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty), mask=row_ok[:, None],
    )
    tl.store(lse_base + row_idx, lse, mask=row_ok)


# seq_bucket is an ordinary integer argument that only keys the autotune cache,
# giving one search per power-of-two length range. Being non-constexpr, it does
# not force a separate compilation per bucket.
_attn_fwd_tuned = triton.autotune(
    configs=forward_configs(),
    key=["HEAD_DIM", "IS_CAUSAL", "seq_bucket"],
    prune_configs_by={"early_config_prune": prune_forward},
)(_attn_fwd)


def attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float | None,
    config: KernelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (o, lse2) where lse2 is the row logsumexp in base 2, shape (B, H, Nq).

    `config` pins the block sizes; otherwise they come from the autotuner, or
    from a fixed default when autotuning is disabled.
    """
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    o = torch.empty((B, H, Nq, D), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, H, Nq), device=q.device, dtype=torch.float32)
    if o.numel() == 0:
        return o, lse
    if Nk == 0:
        # Softmax over an empty set: same convention as a fully masked row.
        o.zero_()
        lse.fill_(float("-inf"))
        return o, lse

    args = (
        q, k, v, o, lse,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        lse.stride(0), lse.stride(1),
        H, Nq, Nk, sm_scale,
        triton.next_power_of_2(max(Nq, Nk)),
    )
    if config is None and autotune_enabled():
        def grid(meta):
            return (triton.cdiv(Nq, meta["BLOCK_M"]), B * H)

        _attn_fwd_tuned[grid](*args, HEAD_DIM=D, IS_CAUSAL=causal)
        return o, lse

    cfg = config or default_forward_config(D)
    grid = (triton.cdiv(Nq, cfg.block_m), B * H)
    _attn_fwd[grid](
        *args,
        HEAD_DIM=D, IS_CAUSAL=causal,
        BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return o, lse
