"""group_sum_valid: counts[k] += number of valid rows of group k, sums[k] += their values
(modulo 2^64).

Each program takes BLOCK rows and sorts the keys (group << LOG_BLOCK | lane), with
null and out-of-range rows given the group g so that they sort last. A segmented
scan over the sorted keys then gives each group's count and sum at its last
element, and only those elements issue the global atomics: one pair of atomics per
distinct group of the block instead of one per row.
"""

import triton
import triton.language as tl

BLOCK = 2048
LOG_BLOCK = 11
NUM_WARPS = 8

TYPES = {
    "group": "*i32",
    "value": "*i64",
    "valid": "*u8",
    "counts": "*i64",
    "sums": "*i64",
    "n": "i64",
    "g": "i64",
}
CONSTANTS = {"BLOCK": BLOCK, "LOG_BLOCK": LOG_BLOCK}


@triton.jit
def _segment_combine(g1, s1, c1, g2, s2, c2):
    same = g1 == g2
    return g2, s2 + tl.where(same, s1, 0), c2 + tl.where(same, c1, 0)


@triton.jit
def kernel(group, value, valid, counts, sums, n, g, BLOCK: tl.constexpr, LOG_BLOCK: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * BLOCK
    lane = tl.arange(0, BLOCK)
    rows = base + lane
    in_range = rows < n
    grp = tl.load(group + rows, mask=in_range, other=0)
    vbyte = tl.load(valid + (rows >> 3), mask=in_range, other=0).to(tl.int32)
    use = in_range & (((vbyte >> (rows & 7).to(tl.int32)) & 1) != 0)
    key = (tl.where(use, grp, g).to(tl.int32) << LOG_BLOCK) | lane
    key = tl.sort(key)
    kgrp = key >> LOG_BLOCK
    live = kgrp < g
    v = tl.load(value + base + (key & (BLOCK - 1)), mask=live, other=0)
    c = live.to(tl.int64)
    _, s, c = tl.associative_scan((kgrp, v, c), 0, _segment_combine)
    following = tl.gather(kgrp, tl.minimum(lane + 1, BLOCK - 1), 0)
    last = live & ((lane == BLOCK - 1) | (following != kgrp))
    tl.atomic_add(counts + kgrp, c, mask=last)
    tl.atomic_add(sums + kgrp, s, mask=last)


def run(group, value, valid, counts, sums, n, g):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](group, value, valid, counts, sums, n, g, BLOCK=BLOCK, LOG_BLOCK=LOG_BLOCK, num_warps=NUM_WARPS)
