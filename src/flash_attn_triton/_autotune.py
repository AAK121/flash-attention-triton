"""Block-size selection: autotune search spaces, shared-memory pruning, defaults.

All three tuned kernels (forward, dK/dV, dQ) share one search space over
BLOCK_M (query tile), BLOCK_N (key tile), num_warps and num_stages. They differ
in which tiles stay resident for the whole program and which are streamed, and
that is what the per-kernel shared-memory estimates below encode.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
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


# Defaults fit in 48 KB of shared memory for every supported head dim, so they
# are valid on any GPU Triton supports.


def default_forward_config(head_dim: int) -> KernelConfig:
    if head_dim >= 128:
        return KernelConfig(64, 32, num_warps=4)
    return KernelConfig(64, 64, num_warps=4)


def default_dkdv_config(head_dim: int) -> KernelConfig:
    if head_dim >= 128:
        return KernelConfig(32, 32, num_warps=2)
    return KernelConfig(32, 64, num_warps=4)


def default_dq_config(head_dim: int) -> KernelConfig:
    if head_dim >= 128:
        return KernelConfig(32, 32, num_warps=2)
    return KernelConfig(64, 32, num_warps=4)


def search_space() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in itertools.product([32, 64, 128], [32, 64, 128], [2, 4, 8], [1, 2, 3])
    ]


# Shared-memory estimates, in elements. Each is conservative: rejecting a config
# that would have fit only loses a candidate, while accepting one that does not
# fit costs a compile that Triton then discards. Matrices that are converted from
# the accumulator layout to an MMA operand layout (P, dS) round-trip through
# shared memory and are counted too.


def smem_forward(bm: int, bn: int, d: int, stages: int) -> int:
    # Q resident; K, V streamed; P converted.
    return bm * d + 2 * bn * d * stages + bm * bn


def smem_dkdv(bm: int, bn: int, d: int, stages: int) -> int:
    # K, V resident; Q, dO streamed; P^T and dS^T converted.
    return 2 * bn * d + 2 * bm * d * stages + 2 * bm * bn


def smem_dq(bm: int, bn: int, d: int, stages: int) -> int:
    # Q, dO resident; K, V streamed; P and dS converted.
    return 2 * bm * d + 2 * bn * d * stages + 2 * bm * bn


def make_pruner(smem_elems: Callable[[int, int, int, int], int], rows_key: str, tensor_arg: str):
    """Build an early_config_prune hook.

    rows_key names the tile dimension that indexes the rows of the kernel's
    accumulator (BLOCK_M for forward and dQ, BLOCK_N for dK/dV).
    """

    def prune(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        d = args["HEAD_DIM"]
        t = args[tensor_arg]
        limit = max_shared_memory(t.device)
        # Multi-stage pipelining of global loads is built on cp.async, which
        # first appeared in sm80. On Turing, extra stages only consume shared
        # memory.
        max_stages = 3 if compute_capability(t.device) >= (8, 0) else 1
        kept = []
        for c in configs:
            bm, bn = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]
            if c.num_stages > max_stages:
                continue
            if smem_elems(bm, bn, d, c.num_stages) * t.element_size() > limit:
                continue
            # With more than one warp per 16 accumulator rows, warps also split
            # the other tile dimension, and every row reduction then needs a
            # cross-warp exchange through shared memory.
            if c.num_warps > c.kwargs[rows_key] // 16:
                continue
            kept.append(c)
        return kept or configs[:1]

    return prune


prune_forward = make_pruner(smem_forward, "BLOCK_M", "q_ptr")
prune_dkdv = make_pruner(smem_dkdv, "BLOCK_N", "k_ptr")
prune_dq = make_pruner(smem_dq, "BLOCK_M", "q_ptr")
