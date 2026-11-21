from __future__ import annotations

import torch

from ._device import bf16_supported, interpreting
from ._fwd import attn_forward

_SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (batch, heads, seq, head_dim)")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(
            "q and k must agree on batch, heads and head_dim "
            f"(got {tuple(q.shape)} and {tuple(k.shape)})"
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


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention without materializing the score matrix.

    Args:
        q: (batch, heads, seq_q, head_dim)
        k, v: (batch, heads, seq_k, head_dim)
        causal: when True, query i attends keys j <= i + (seq_k - seq_q).
        sm_scale: logit scale, defaults to head_dim ** -0.5.

    Inputs may be arbitrarily strided. Output is contiguous with q's shape.
    """
    _check_inputs(q, k, v)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError("the backward pass is not implemented yet")
    o, _ = attn_forward(q, k, v, causal, sm_scale)
    return o
