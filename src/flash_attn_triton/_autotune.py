"""Block-size selection: autotune search space, shared-memory pruning, defaults."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

import triton

from ._device import compute_capability, interpreting, max_shared_memory


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_warps: int = 4
    num_stages: int = 1


def autotune_enabled() -> bool:
    """Autotuning is on by default; FLASH_ATTN_TRITON_AUTOTUNE=0 uses fixed defaults.

    The test suite turns it off so that hundreds of shapes do not each pay for a
    search, and exercises block sizes explicitly instead.
    """
    return not interpreting() and os.environ.get("FLASH_ATTN_TRITON_AUTOTUNE", "1") != "0"


def default_forward_config(head_dim: int) -> KernelConfig:
    # Fits in 48 KB of shared memory for every supported head dim, so it is
    # valid on any GPU Triton supports.
    if head_dim >= 128:
        return KernelConfig(64, 32, num_warps=4, num_stages=1)
    return KernelConfig(64, 64, num_warps=4, num_stages=1)


def forward_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in itertools.product([32, 64, 128], [32, 64, 128], [2, 4, 8], [1, 2, 3])
    ]


def estimate_forward_smem(block_m: int, block_n: int, head_dim: int, stages: int, elem: int) -> int:
    """Conservative shared-memory estimate for one forward program.

    Q is staged once, K and V are buffered `stages` deep by the software
    pipeline, and P is round-tripped through shared memory to convert it from
    the accumulator layout to an MMA operand layout. Rejecting a config that
    would have fit only loses a candidate; accepting one that does not fit costs
    a compile that Triton then discards.
    """
    q = block_m * head_dim
    kv = 2 * block_n * head_dim * max(stages, 1)
    p = block_m * block_n
    return (q + kv + p) * elem


def prune_forward(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    head_dim = args["HEAD_DIM"]
    q = args["q_ptr"]
    limit = max_shared_memory(q.device)
    # Multi-stage pipelining of global loads is built on cp.async, which first
    # appeared in sm80. On Turing, extra stages only consume shared memory.
    max_stages = 3 if compute_capability(q.device) >= (8, 0) else 1
    kept = []
    for c in configs:
        bm, bn = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]
        if c.num_stages > max_stages:
            continue
        if estimate_forward_smem(bm, bn, head_dim, c.num_stages, q.element_size()) > limit:
            continue
        # With more than one warp per 16 query rows, warps also split the key
        # dimension of S, and the row max and row sum then need a cross-warp
        # reduction through shared memory on every iteration.
        if c.num_warps > bm // 16:
            continue
        kept.append(c)
    return kept or configs[:1]
