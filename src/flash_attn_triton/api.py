from __future__ import annotations

import torch

from ._bwd import attn_backward, attn_backward_varlen
from ._device import bf16_supported, interpreting, warn_if_dot_lacks_tensor_cores
from ._fwd import attn_forward, attn_forward_varlen

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
    if q.device.type == "cuda":
        warn_if_dot_lacks_tensor_cores(q.device)


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


class FlashAttentionVarlenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, causal, sm_scale, window):
        o, lse = attn_forward_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal, sm_scale, window)
        ctx.save_for_backward(q, k, v, o, lse, cu_q, cu_k)
        ctx.args = (max_q, max_k, causal, sm_scale, window)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu_q, cu_k = ctx.saved_tensors
        max_q, max_k, causal, sm_scale, window = ctx.args
        dq, dk, dv = attn_backward_varlen(
            do, q, k, v, o, lse, cu_q, cu_k, max_q, max_k, causal, sm_scale, window
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    sm_scale: float | None = None,
    window_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Attention over a packed batch of sequences with different lengths.

    Args:
        q: (total_q, heads, head_dim), all query sequences concatenated.
        k, v: (total_k, kv_heads, head_dim).
        cu_seqlens_q, cu_seqlens_k: int32 tensors of length batch + 1 with the
            running offsets; sequence b is q[cu_seqlens_q[b]:cu_seqlens_q[b + 1]].
        max_seqlen_q, max_seqlen_k: longest sequence in the batch; sizes the grid.
        causal, sm_scale, window_size: as in flash_attention, per sequence.

    No padding is stored or computed: programs beyond a sequence's length exit
    immediately.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed q, k, v must have shape (tokens, heads, head_dim)")
    # Reuse the dense checks on (1, heads, tokens, dim) views of the packed tensors.
    _check_inputs(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)))
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.dtype != torch.int32 or cu.ndim != 1 or cu.device != q.device:
            raise ValueError(f"{name} must be a 1-D int32 tensor on the inputs' device")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")
    window = _check_window(window_size)
    return FlashAttentionVarlenFunction.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k),
        causal, sm_scale, window,
    )
