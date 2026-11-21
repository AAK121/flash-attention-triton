"""Attention throughput and memory sweep.

Compares, in fp16 on one GPU:
    naive        attention_reference with fp16 compute (materializes N x N)
    sdpa_math    torch SDPA forced to the MATH backend
    sdpa_mem     torch SDPA forced to the EFFICIENT_ATTENTION backend
    sdpa_flash   torch SDPA forced to the FLASH_ATTENTION backend (sm80+ only)
    triton       this package

Workload follows the FlashAttention-2 paper: batch * seq = 16k tokens, model
width 2048, heads = 2048 / head_dim. For each (mode, head_dim, causal) group
the sequence length grows from 256 to 16k; once a provider runs out of memory
at some length it is recorded as OOM for every longer length in that group
without being run again.

FLOPs use the standard count: 4 * B * H * Nq * Nk * D for the forward (two
matmuls), halved under a causal mask, 2.5x that for the backward (five
matmuls), 3.5x for forward + backward. The count is the same for every
provider, so TFLOP/s compares wall time on a common scale; it does not credit
or penalize recomputation inside a kernel.

Every point is measured with triton.testing.do_bench in REPEATS independent
calls; the median of the per-call medians is reported together with the
spread across calls.

    python benchmarks/bench_attention.py --out results/run.csv [--shard 0 --num-shards 2]
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.runtime.errors import OutOfResources

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_info  # noqa: E402
from flash_attn_triton import attention_reference, flash_attention  # noqa: E402

SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384]
HEAD_DIMS = [64, 128]
MODES = ["fwd", "bwd", "fwd_bwd"]
TOKENS = 16384
WIDTH = 2048
REPEATS = 5
WARMUP_MS = 25
REP_MS = 100

PROVIDERS = {
    "naive": None,
    "sdpa_math": SDPBackend.MATH,
    "sdpa_mem": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "triton": None,
}


@dataclass
class Row:
    provider: str
    mode: str
    head_dim: int
    causal: bool
    seq_len: int
    batch: int
    heads: int
    status: str
    ms_median: float = math.nan
    ms_p20: float = math.nan
    ms_p80: float = math.nan
    ms_std_across_repeats: float = math.nan
    tflops: float = math.nan
    pct_peak: float = math.nan
    peak_mem_mb: float = math.nan
    extra_mem_mb: float = math.nan


def attention_fn(provider: str, causal: bool):
    if provider == "triton":
        return lambda q, k, v: flash_attention(q, k, v, causal=causal)
    if provider == "naive":
        return lambda q, k, v: attention_reference(
            q, k, v, causal=causal, compute_dtype=torch.float16
        )
    backend = PROVIDERS[provider]

    def run(q, k, v):
        with sdpa_kernel(backend):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return run


def provider_available(provider: str) -> bool:
    if provider == "sdpa_flash":
        # PyTorch's flash backend requires sm80+; on Turing it would only raise.
        return torch.cuda.get_device_capability() >= (8, 0)
    return True


def flops(mode: str, b: int, h: int, n: int, d: int, causal: bool) -> float:
    f = 4.0 * b * h * n * n * d
    if causal:
        f /= 2
    return f * {"fwd": 1.0, "bwd": 2.5, "fwd_bwd": 3.5}[mode]


def build_step(provider: str, mode: str, b, h, n, d, causal, device):
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd():
        return torch.randn(b, h, n, d, generator=g).to(device=device, dtype=torch.float16)

    needs_grad = mode != "fwd"
    q, k, v = (rnd().requires_grad_(needs_grad) for _ in range(3))
    do = rnd()
    fn = attention_fn(provider, causal)

    if mode == "fwd":
        def step():
            with torch.no_grad():
                fn(q, k, v)
        return step, [q, k, v]

    if mode == "bwd":
        o = fn(q, k, v)

        def step():
            o.backward(do, retain_graph=True)
        return step, [q, k, v]

    def step():
        fn(q, k, v).backward(do)
    return step, [q, k, v]


def measure(provider, mode, d, causal, n, device, peak) -> Row:
    b = max(1, TOKENS // n)
    h = WIDTH // d
    row = Row(provider, mode, d, causal, n, b, h, status="ok")
    torch.cuda.empty_cache()
    gc.collect()
    try:
        base = torch.cuda.memory_allocated()
        step, leaves = build_step(provider, mode, b, h, n, d, causal, device)
        inputs = torch.cuda.memory_allocated()
        # First call outside the timed region: compiles and autotunes.
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        row.peak_mem_mb = (peak_mem - base) / 2**20
        row.extra_mem_mb = (peak_mem - inputs) / 2**20

        medians, p20s, p80s = [], [], []
        for _ in range(REPEATS):
            p20, p50, p80 = triton.testing.do_bench(
                step, warmup=WARMUP_MS, rep=REP_MS, grad_to_none=leaves,
                quantiles=[0.2, 0.5, 0.8],
            )
            medians.append(p50)
            p20s.append(p20)
            p80s.append(p80)
        row.ms_median = statistics.median(medians)
        row.ms_p20 = statistics.median(p20s)
        row.ms_p80 = statistics.median(p80s)
        row.ms_std_across_repeats = statistics.stdev(medians) if len(medians) > 1 else 0.0
        row.tflops = flops(mode, b, h, n, d, causal) / (row.ms_median * 1e-3) / 1e12
        if peak:
            row.pct_peak = 100.0 * row.tflops / peak
    except torch.cuda.OutOfMemoryError:
        row.status = "oom"
    except OutOfResources:
        # A kernel config that does not fit in shared memory or registers. Not
        # expected with pruning in place; recorded rather than aborting the sweep.
        row.status = "oor"
    finally:
        torch.cuda.empty_cache()
    return row


def shard_groups(shard: int, num_shards: int):
    """(mode, head_dim, causal) groups for one shard.

    Sharding is by (head_dim, causal) so that each kernel variant is compiled
    and autotuned in exactly one process; all three modes of a variant share
    the forward kernels. Variants are dealt out in snake order (0 1 1 0 ...),
    which pairs the cheapest with the most expensive when there are two GPUs.
    """
    variants = [(d, c) for d in HEAD_DIMS for c in (False, True)]
    order = []
    for i, v in enumerate(variants):
        lap, pos = divmod(i, num_shards)
        order.append((pos if lap % 2 == 0 else num_shards - 1 - pos, v))
    mine = [v for owner, v in order if owner == shard]
    return [(m, d, c) for d, c in mine for m in MODES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--providers", default=",".join(PROVIDERS))
    ap.add_argument("--max-seq", type=int, default=max(SEQ_LENS))
    a = ap.parse_args()

    device = torch.device("cuda")
    providers = [p for p in a.providers.split(",") if provider_available(p)]
    peak = env_info.peak_tflops(torch.cuda.get_device_name())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    env_info.write(
        a.out.replace(".csv", ".meta.json"),
        {"shard": a.shard, "num_shards": a.num_shards, "providers": providers,
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
         "repeats": REPEATS, "warmup_ms": WARMUP_MS, "rep_ms": REP_MS,
         "tokens_per_batch": TOKENS, "model_width": WIDTH},
    )

    fields = list(Row.__dataclass_fields__)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode, d, causal in shard_groups(a.shard, a.num_shards):
            for provider in providers:
                oom = False
                for n in [s for s in SEQ_LENS if s <= a.max_seq]:
                    if oom:
                        row = Row(provider, mode, d, causal, n, max(1, TOKENS // n),
                                  WIDTH // d, status="oom")
                    else:
                        t0 = time.time()
                        row = measure(provider, mode, d, causal, n, device, peak)
                        oom = row.status == "oom"
                        print(
                            f"[shard {a.shard}] {mode:7s} d={d:3d} causal={int(causal)} "
                            f"n={n:5d} {provider:10s} {row.status:3s} "
                            f"{row.ms_median:9.3f} ms {row.tflops:6.2f} TFLOP/s "
                            f"({time.time() - t0:.0f}s)",
                            flush=True,
                        )
                    w.writerow(asdict(row))
                    f.flush()


if __name__ == "__main__":
    main()
