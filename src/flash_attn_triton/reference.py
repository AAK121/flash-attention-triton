"""Naive attention that materializes the full score matrix.

Used as the correctness oracle. Computation is done in fp32 regardless of the
input dtype so that it can serve as ground truth for fp16/bf16 kernels.
"""

from __future__ import annotations

import math

import torch


def causal_mask(nq: int, nk: int, device: torch.device) -> torch.Tensor:
    """True where query i may attend key j.

    Queries are aligned to the end of the key sequence (bottom-right), so with
    nq < nk the last query sees every key. This matches decoding with a KV cache
    and the convention used by the reference FlashAttention implementation.
    """
    i = torch.arange(nq, device=device)[:, None]
    j = torch.arange(nk, device=device)[None, :]
    return j <= i + (nk - nq)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    return_lse: bool = False,
):
    """Softmax(q k^T * sm_scale) v for tensors shaped (batch, heads, seq, head_dim).

    Rows that have no visible key (possible when causal and nq > nk) produce a
    zero output and a logsumexp of -inf.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    qf, kf, vf = (t.to(compute_dtype) for t in (q, k, v))
    s = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    if causal:
        mask = causal_mask(q.shape[-2], k.shape[-2], q.device)
        s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s.float(), dim=-1)
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.matmul(p, vf)
    if return_lse:
        return o, lse
    return o
