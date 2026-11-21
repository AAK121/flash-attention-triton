from __future__ import annotations

import torch

from ._bwd import attn_backward
from ._device import bf16_supported, interpreting
from ._fwd import attn_forward

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (batch, heads, seq, head_dim)")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(
            "q and k must agree on batch and head_dim "
            f"(got {tuple(q.shape)} and {tuple(k.shape)})"
        )
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"query heads ({q.shape[1]}) must be a multiple of key/value heads ({k.shape[1]})"
        )
    if q.shape[3] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {_SUPPORTED_HEAD_DIMS}, got {q.shape[3]}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must share a dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}; use float16 or bfloat16")
    if q.dtype == torch.bfloat16 and not bf16_supported(q.device):
        raise ValueError("bfloat16 needs compute capability 8.0 or newer")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")
    if q.device.type != "cuda" and not interpreting():
        raise ValueError("inputs must be CUDA tensors (or set TRITON_INTERPRET=1 for CPU)")


def _check_window(window_size: tuple[int, int] | None) -> tuple[int, int] | None:
    if window_size is None:
        return None
    left, right = (int(x) for x in window_size)
    if left < -1 or right < -1:
        raise ValueError("window_size entries must be >= 0, or -1 for unbounded")
    return (left, right)


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, window):
        o, lse = attn_forward(q, k, v, causal, sm_scale, window)
        # Only O and the per-row logsumexp are kept for the backward pass:
        # O(N) extra memory instead of the O(N^2) attention matrix.
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.window = window
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = attn_backward(
            do, q, k, v, o, lse, ctx.causal, ctx.sm_scale, ctx.window
        )
        return dq, dk, dv, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    window_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention without materializing the score matrix.

    Args:
        q: (batch, heads, seq_q, head_dim)
        k, v: (batch, kv_heads, seq_k, head_dim). kv_heads must divide heads;
            kv_heads < heads gives grouped-query attention, kv_heads == 1
            multi-query attention.
        causal: when True, query i attends keys j <= i + (seq_k - seq_q).
        sm_scale: logit scale, defaults to head_dim ** -0.5.
        window_size: (left, right) for local attention: query i attends keys
            within [i + off - left, i + off + right], off = seq_k - seq_q. -1
            leaves that side unbounded. Combined with causal, right is 0.

    Inputs may be arbitrarily strided. The output is contiguous with q's shape
    and is differentiable with respect to q, k and v.
    """
    _check_inputs(q, k, v)
    window = _check_window(window_size)
    return FlashAttentionFunction.apply(q, k, v, causal, sm_scale, window)
