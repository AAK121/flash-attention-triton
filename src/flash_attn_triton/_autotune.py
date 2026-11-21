"""Autotune search spaces and shared-memory pruning."""

from __future__ import annotations

import itertools

import triton

from ._device import interpreting, max_shared_memory


def _configs(block_ms, block_ns, warps, stages):
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in itertools.product(block_ms, block_ns, warps, stages)
    ]


def forward_configs():
    if interpreting():
        return _configs([32], [32], [1], [1])
    return _configs([32, 64, 128], [32, 64, 128], [2, 4, 8], [1, 2, 3])


def estimate_forward_smem(block_m: int, block_n: int, head_dim: int, stages: int, elem: int) -> int:
    """Upper-bound estimate of shared memory for one program.

    The Q tile is staged once, K and V tiles are multi-buffered by the software
    pipeline. The accumulator and softmax statistics live in registers. The
    estimate is deliberately conservative: a config that is rejected here but
    would have fit only costs a missed candidate, whereas one that is accepted
    and does not fit aborts the launch.
    """
    q = block_m * head_dim * elem
    kv = 2 * block_n * head_dim * elem * max(stages, 1)
    return q + kv


def prune_forward(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    head_dim = args["HEAD_DIM"]
    elem = args["q_ptr"].element_size()
    limit = max_shared_memory(args["q_ptr"].device)
    kept = []
    for c in configs:
        bm, bn = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]
        if estimate_forward_smem(bm, bn, head_dim, c.num_stages, elem) > limit:
            continue
        # The MMA layout assigns at least 16 query rows to a warp; more warps
        # than that leave warps idle and only add synchronization cost.
        if c.num_warps > max(1, bm // 16) * 2:
            continue
        kept.append(c)
    return kept or configs[:1]
