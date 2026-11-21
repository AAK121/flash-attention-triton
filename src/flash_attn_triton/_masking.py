"""Visibility rules shared by the forward and backward kernels.

Query i may attend key j when

    i + off - left  <=  j  <=  i + off + right,      off = Nk - Nq,

where either bound can be absent. `off` aligns the last query with the last
key (bottom-right), which is what decoding with a KV cache needs. Causal
attention is the special case right = 0 with no left bound; sliding-window
attention adds a finite left bound.

The forward and dQ kernels own a block of queries and scan keys. The dK/dV
kernel owns a block of keys and scans queries. Solving the same inequality for
i instead of j gives

    j - off - right  <=  i  <=  j - off + left,

which is the first form with off -> -off and left <-> right exchanged, so one
range helper serves both directions.
"""

import triton
import triton.language as tl


@triton.jit
def scan_range(
    first,
    n_other,
    off,
    left,
    right,
    BLOCK_SELF: tl.constexpr,
    BLOCK_OTHER: tl.constexpr,
    LIMIT_LEFT: tl.constexpr,
    LIMIT_RIGHT: tl.constexpr,
):
    """Tile bounds for the program owning indices [first, first + BLOCK_SELF).

    Returns (lo, full_lo, full_hi, hi) over the other axis:
      [lo, full_lo)       partially visible tiles, need the elementwise mask
      [full_lo, full_hi)  tiles visible to every owned index, no mask needed
      [full_hi, hi)       partially visible tiles, or the ragged end of the axis
    Everything outside [lo, hi) is invisible to the whole block and skipped.
    lo and full_hi are multiples of BLOCK_OTHER, so loops starting there stay
    tile aligned.
    """
    last = first + BLOCK_SELF - 1
    if LIMIT_LEFT:
        lo = tl.maximum(first + off - left, 0)
        full_lo = tl.cdiv(tl.maximum(last + off - left, 0), BLOCK_OTHER) * BLOCK_OTHER
    else:
        lo = 0
        full_lo = 0
    if LIMIT_RIGHT:
        hi = tl.minimum(last + off + right + 1, n_other)
        full_hi = (tl.minimum(first + off + right + 1, n_other) // BLOCK_OTHER) * BLOCK_OTHER
    else:
        hi = n_other
        full_hi = (n_other // BLOCK_OTHER) * BLOCK_OTHER
    hi = tl.maximum(hi, 0)
    lo = tl.minimum((lo // BLOCK_OTHER) * BLOCK_OTHER, hi)
    full_lo = tl.minimum(tl.maximum(full_lo, lo), hi)
    full_hi = tl.minimum(tl.maximum(full_hi, full_lo), hi)
    return lo, full_lo, full_hi, hi


@triton.jit
def window_mask(
    keep,
    i,
    j,
    off,
    left,
    right,
    LIMIT_LEFT: tl.constexpr,
    LIMIT_RIGHT: tl.constexpr,
):
    """AND the window constraint into `keep`; i and j are broadcastable index tiles."""
    if LIMIT_LEFT:
        keep = keep & (j >= i + off - left)
    if LIMIT_RIGHT:
        keep = keep & (j <= i + off + right)
    return keep
