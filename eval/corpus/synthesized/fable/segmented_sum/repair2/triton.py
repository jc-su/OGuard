"""segmented_sum: sums[j] = sum of values[offsets[j]:offsets[j + 1]].

The kernel picks one of two paths per launch from the average segment length
(offsets[g] / g):

- short segments (average below WARP_MIN_AVG values): a program handles
  BLOCK_G segments; one round over a [BLOCK_G, VALUES_A] tile reads VALUES_A
  values of every segment, and the rounds run up to the longest segment of
  the tile;
- long segments: programs walk the segments SEGMENTS_B at a time with a grid
  stride and a [SEGMENTS_B, VALUES_B] tile, so each round reads VALUES_B
  consecutive values of a segment with coalesced loads, and a few very long
  segments cannot stall a whole tile of short ones.

Both paths accumulate in the tile and reduce once per segment. The kernel
picks the path from offsets[g]; run picks the grid for that path from the
length of values, which the host knows without a device sync. Both paths are
correct on any grid, so a mismatch only costs speed.

Value positions are the 32-bit sum offsets[s] + j taken as unsigned, and a
value is read only when that position is below offsets[s + 1], compared as
unsigned. Under the contract (nondecreasing offsets with offsets[g] = n) this
is exactly j < offsets[s + 1] - offsets[s]: a sum that wraps past INT_MAX
compares above offsets[s + 1] and is skipped, and such a j is past the end
of the segment. The position is zero-extended into the address, so it is
non-negative by construction and below offsets[s + 1], the length of values
at most, on every load.
"""

import triton
import triton.language as tl

BLOCK_G = 32
VALUES_A = 4
SEGMENTS_B = 4
VALUES_B = 32
WARP_MIN_AVG = 32
NUM_WARPS = 4

TYPES = {"offsets": "*i32", "values": "*fp64", "sums": "*fp64", "g": "i64"}
CONSTANTS = {
    "BLOCK_G": BLOCK_G,
    "VALUES_A": VALUES_A,
    "SEGMENTS_B": SEGMENTS_B,
    "VALUES_B": VALUES_B,
    "WARP_MIN_AVG": WARP_MIN_AVG,
}


@triton.jit
def sum_segments(offsets, values, sums, s, valid, S: tl.constexpr, V: tl.constexpr):
    begin = tl.load(offsets + s, mask=valid, other=0)
    end = tl.load(offsets + s + 1, mask=valid, other=0)
    longest = tl.max(end - begin, axis=0)
    end_u = end.to(tl.uint32)[:, None]
    acc = tl.zeros([S, V], tl.float64)
    for j0 in range(0, longest, V):
        j = j0 + tl.arange(0, V)
        p = (begin[:, None] + j[None, :]).to(tl.uint32)
        in_segment = p < end_u
        acc += tl.load(values + p.to(tl.int64), mask=in_segment, other=0.0)
    tl.store(sums + s, tl.sum(acc, axis=1), mask=valid)


@triton.jit
def kernel(
    offsets,
    values,
    sums,
    g: tl.int64,
    BLOCK_G: tl.constexpr,
    VALUES_A: tl.constexpr,
    SEGMENTS_B: tl.constexpr,
    VALUES_B: tl.constexpr,
    WARP_MIN_AVG: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n = tl.load(offsets + g)
    if n >= g * WARP_MIN_AVG:
        stride = tl.num_programs(0).to(tl.int64) * SEGMENTS_B
        for s0 in range(pid * SEGMENTS_B, g, stride):
            s = s0 + tl.arange(0, SEGMENTS_B)
            sum_segments(offsets, values, sums, s, s < g, SEGMENTS_B, VALUES_B)
    else:
        s = pid * BLOCK_G + tl.arange(0, BLOCK_G)
        sum_segments(offsets, values, sums, s, s < g, BLOCK_G, VALUES_A)


def run(offsets, values, sums, g):
    wide = values.numel() >= g * WARP_MIN_AVG
    grid = (max(1, triton.cdiv(g, SEGMENTS_B if wide else BLOCK_G)),)
    kernel[grid](offsets, values, sums, g, **CONSTANTS, num_warps=NUM_WARPS)
