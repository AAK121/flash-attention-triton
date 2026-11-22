# ruff: noqa: B023, B007
"""Inspect what the compiler produced for the attention kernels on this GPU.

Prints, per compiled kernel variant, the register count, spill count and the
number of tensor-core (mma) instructions in the PTX, plus a reference Triton
matmul so the attention numbers can be read against what tl.dot reaches on its
own. Used to explain the benchmark results, not as a benchmark itself.

    python benchmarks/inspect_kernels.py
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("FLASH_ATTN_TRITON_AUTOTUNE", "0")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flash_attn_triton import _bwd, _fwd  # noqa: E402
from flash_attn_triton._autotune import KernelConfig  # noqa: E402
from flash_attn_triton.api import flash_attention  # noqa: E402


@triton.jit
def _matmul(a_ptr, b_ptr, c_ptr, M, N, K,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def describe(kernel) -> str:
    asm = getattr(kernel, "asm", {}) or {}
    ptx = asm.get("ptx", "")
    mma = len(re.findall(r"\bmma\.sync", ptx))
    fma = len(re.findall(r"\bfma\.rn\.f(16|32)", ptx))
    local = len(re.findall(r"\b(ld|st)\.local", ptx))
    return (f"regs={getattr(kernel, 'n_regs', '?')} spills={getattr(kernel, 'n_spills', '?')} "
            f"ptx: mma.sync={mma} fma={fma} local_ld_st={local}")


def compiled(fn):
    out = []
    # Triton 3.2 keeps fn.cache[device]; later releases fn.device_caches[device][0].
    caches = getattr(fn, "device_caches", None) or getattr(fn, "cache", {})
    for entry in caches.values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for key, kernel in cache.items():
            out.append((str(key)[:160], kernel))
    return out


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main() -> None:
    dev = torch.cuda.get_device_name()
    print(f"device {dev} cc {torch.cuda.get_device_capability()} "
          f"triton {triton.__version__} torch {torch.__version__}")

    n = 2048
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    flops = 2 * n**3
    ms_torch = bench(lambda: torch.mm(a, b))
    print(f"torch.mm {n}^3: {flops / ms_torch / 1e9:.2f} TFLOP/s")
    for bm, bn, bk, w in [(64, 64, 32, 4), (128, 128, 32, 8), (128, 64, 32, 4)]:
        grid = (n // bm, n // bn)
        _matmul[grid](a, b, c, n, n, n, BM=bm, BN=bn, BK=bk, num_warps=w)
        ms = bench(lambda: _matmul[grid](a, b, c, n, n, n, BM=bm, BN=bn, BK=bk, num_warps=w))
        print(f"triton matmul BM={bm} BN={bn} BK={bk} w={w}: {flops / ms / 1e9:.2f} TFLOP/s")
    torch.testing.assert_close(c.float(), (a.float() @ b.float()), atol=1.0, rtol=1e-2)
    for key, k in compiled(_matmul):
        print(f"  matmul variant: {describe(k)}")

    # The benchmark shape at N=1024: 16k tokens, width 2048.
    for d in (64, 128):
        bsz, h, seq = 16, 2048 // d, 1024
        q, k_, v = (torch.randn(bsz, h, seq, d, device="cuda", dtype=torch.float16)
                    for _ in range(3))
        fl = 4 * bsz * h * seq * seq * d
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k_, v))
        print(f"\nd={d} sdpa_mem fwd: {ms:.2f} ms {fl / ms / 1e9:.2f} TFLOP/s")
        for bm, bn, w in [(64, 32, 4), (64, 64, 4), (128, 32, 8), (128, 64, 8), (32, 32, 2)]:
            cfg = KernelConfig(bm, bn, num_warps=w)
            try:
                _fwd.attn_forward(q, k_, v, False, None, config=cfg)
                ms = bench(lambda cfg=cfg: _fwd.attn_forward(q, k_, v, False, None, config=cfg))
                print(f"d={d} triton fwd BM={bm} BN={bn} w={w}: {ms:.2f} ms "
                      f"{fl / ms / 1e9:.2f} TFLOP/s")
            except Exception as e:  # report and keep going, this is a diagnostic
                print(f"d={d} BM={bm} BN={bn} w={w}: {type(e).__name__} {str(e)[:120]}")
        qg, kg, vg = (t.clone().requires_grad_() for t in (q, k_, v))
        o = flash_attention(qg, kg, vg)
        do = torch.randn_like(o)
        ms = bench(lambda: torch.autograd.grad(o, (qg, kg, vg), do, retain_graph=True))
        print(f"d={d} triton bwd (default configs): {ms:.2f} ms "
              f"{2.5 * fl / ms / 1e9:.2f} TFLOP/s")

    print("\ncompiled forward variants:")
    for key, k in compiled(_fwd._attn_fwd):
        print(f"  {describe(k)}")
    print("compiled backward variants:")
    for fn in (_bwd._attn_bwd_dkdv, _bwd._attn_bwd_dq):
        for key, k in compiled(fn):
            print(f"  {fn.__name__}: {describe(k)}")


if __name__ == "__main__":
    main()
