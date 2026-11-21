from __future__ import annotations

import torch

from flash_attn_triton import attention_reference

# Half-ulp relative rounding error of the output format: the smallest error any
# kernel can be expected to reach when it rounds its result to dtype.
EPS = {torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8}


def make_qkv(b, h, nq, nk, d, dtype, device, seed=0, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)

    return rnd(b, h, nq, d), rnd(b, h, nk, d), rnd(b, h, nk, d)


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def assert_close_to_reference(out, ref32, torch_low, dtype, what, slack=2.0, scale=None):
    """Principled tolerance.

    `torch_low` is the same naive computation executed in the low-precision
    dtype. Its distance from the fp32 reference measures how much error the
    format itself introduces. The kernel is allowed `slack` times that error,
    plus one rounding step (EPS * scale) so that cases where the low-precision
    baseline happens to be exact do not demand exactness. `scale` defaults to
    max|ref|; pass the magnitude of the terms being summed when the result can
    cancel to zero.
    """
    if scale is None:
        scale = ref32.abs().max().item()
    floor = EPS[dtype] * scale
    budget = slack * max_err(torch_low, ref32) + floor
    err = max_err(out, ref32)
    assert err <= budget, f"{what}: error {err:.3e} exceeds budget {budget:.3e}"


def low_precision_reference(q, k, v, causal, sm_scale=None):
    return attention_reference(q, k, v, causal=causal, sm_scale=sm_scale, compute_dtype=q.dtype)


def reference_grads(q, k, v, do, causal, sm_scale=None, compute_dtype=torch.float32):
    """(o, [dq, dk, dv]) of the naive implementation computed in compute_dtype."""
    leaves = [t.detach().to(compute_dtype).requires_grad_() for t in (q, k, v)]
    o = attention_reference(
        *leaves, causal=causal, sm_scale=sm_scale, compute_dtype=compute_dtype
    )
    o.backward(do.to(compute_dtype))
    return o.detach(), [t.grad for t in leaves]


def kernel_grads(fn, q, k, v, do, **kwargs):
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    o = fn(*leaves, **kwargs)
    o.backward(do)
    return o.detach(), [t.grad for t in leaves]


def assert_grads_close(q, k, v, do, causal, grads, dtype, what, sm_scale=None):
    _, ref = reference_grads(q, k, v, do, causal, sm_scale)
    _, low = reference_grads(q, k, v, do, causal, sm_scale, compute_dtype=dtype)
    # dQ and dK are sums of P * (dP - delta) terms that cancel exactly in some
    # cases (a single visible key gives dQ = dK = 0), so their rounding floor is
    # set by the gradient scale of the whole problem rather than by their own
    # maximum.
    scale = max(r.abs().max().item() for r in ref)
    for name, g, r, lo in zip(("dq", "dk", "dv"), grads, ref, low):
        assert g.dtype == dtype
        assert torch.isfinite(g.float()).all(), f"{what}: non-finite {name}"
        assert_close_to_reference(g, r, lo, dtype, f"{what} {name}", scale=scale)
