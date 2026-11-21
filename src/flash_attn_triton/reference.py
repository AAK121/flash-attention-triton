"""Naive attention that materializes the full score matrix.

Used as the correctness oracle. Computation is done in fp32 regardless of the
input dtype so that it can serve as ground truth for fp16/bf16 kernels.
"""

from __future__ import annotations

import math

import torch


def visible_mask(
    nq: int,
    nk: int,
    causal: bool,
    window: tuple[int, int] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """True where query i may attend key j, or None when every key is visible.

    Queries are aligned to the end of the key sequence (bottom-right), so with
    nq < nk the last query sees every key. With window = (left, right), query i
    sees keys in [i + off - left, i + off + right], off = nk - nq; a negative
    bound is unbounded. causal sets right = 0.
    """
    left, right = window if window is not None else (-1, -1)
    if causal:
        right = 0
    if left < 0 and right < 0:
        return None
    i = torch.arange(nq, device=device)[:, None]
    j = torch.arange(nk, device=device)[None, :]
    diag = i + (nk - nq)
    mask = torch.ones(nq, nk, dtype=torch.bool, device=device)
    if left >= 0:
        mask &= j >= diag - left
    if right >= 0:
        mask &= j <= diag + right
    return mask


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    window_size: tuple[int, int] | None = None,
    compute_dtype: torch.dtype = torch.float32,
    return_lse: bool = False,
):
    """Softmax(q k^T * sm_scale) v for q (batch, heads, seq, dim), k/v (batch, kv_heads, seq, dim).

    kv_heads must divide heads; each group of heads // kv_heads query heads
    shares one key/value head. Rows with no visible key produce a zero output
    and a logsumexp of -inf.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    group = q.shape[1] // k.shape[1]
    qf, kf, vf = (t.to(compute_dtype) for t in (q, k, v))
    if group > 1:
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
    s = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    mask = visible_mask(q.shape[-2], k.shape[-2], causal, window_size, q.device)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s.float(), dim=-1)
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.matmul(p, vf)
    if return_lse:
        return o, lse
    return o
