"""Trailing rolling sum over a fixed window of 1 to 1024 rows.

Van Herk / Gil-Werman scheme: cut the rows into blocks of `window` rows aligned
at multiples of `window`. The window [i - window + 1, i] then covers the suffix
of one block and the prefix of the next, so
    out[i] = suffix_sum[i - window + 1] + prefix_sum[i],
where both sums run inside a block; when i - window + 1 starts a block (or is
before row 0) the window is prefix_sum[i] alone. Every output is a sum of at
most `window` inputs and no partial sum is subtracted, so the rounding error is
that of summing the window directly.

Each program loads a region of RUN * LANES rows that starts window - 1 rows
before its outputs. The RTX A6000 (GA102) has two float64 units per SM, 1/64 of
its float32 rate, and float64 additions bound this kernel, so the scans are
laid out to add as few times as possible: the region is held as a [RUN, LANES]
tile in which column l is the run of rows l * RUN .. l * RUN + RUN - 1 and
lives in one thread. Each segmented scan (prefix and suffix inside blocks) runs
sequentially down the columns, then once across the LANES column totals, and
the column carry is added back. The suffix sums are moved window - 1 rows
forward with an in-register gather.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _segmented_add(value_a, flag_a, value_b, flag_b):
    # flag_b marks a block boundary at b: the running sum restarts there.
    return tl.where(flag_b, value_b, value_a + value_b), flag_a | flag_b


@triton.jit
def _row(x, row, r):
    """Row r of a [RUN, LANES] float64 tile, selected without float64 additions."""
    bits = tl.where(row == r, x.to(tl.int64, bitcast=True), 0)
    return tl.sum(bits, axis=0).to(tl.float64, bitcast=True)


@triton.jit
def kernel(values, out, n, window, RUN: tl.constexpr, LANES: tl.constexpr):
    REGION: tl.constexpr = RUN * LANES
    w = tl.cast(window, tl.int64)
    halo = w - 1
    g0 = tl.program_id(0).to(tl.int64) * (REGION - halo) - halo
    row = tl.arange(0, RUN)[:, None]
    lane = tl.arange(0, LANES)
    # Multiplying by a run-time 1 hides the contiguity of a column from Triton's
    # coalescing pass, which then keeps each column in one thread.
    runtime_one = (w // w).to(tl.int32)
    t = lane[None, :] * RUN + row * runtime_one
    g = g0 + t
    x = tl.load(values + g, mask=(g >= 0) & (g < n), other=0.0)
    w32 = w.to(tl.int32)
    phase = (((g0 % w + w) % w).to(tl.int32) + t) % w32
    block_start = phase == 0
    block_end = phase == w32 - 1

    # Prefix sums inside blocks: down each column, then across column totals.
    prefix, started = tl.associative_scan((x, block_start), 0, _segmented_add)
    column_total, _ = tl.associative_scan(
        (_row(prefix, row, RUN - 1), tl.max(block_start.to(tl.int32), axis=0) > 0),
        0,
        _segmented_add,
    )
    carry = tl.where(lane > 0, tl.gather(column_total, tl.maximum(lane - 1, 0), 0), 0.0)
    prefix = tl.where(started, prefix, prefix + carry[None, :])

    # Suffix sums inside blocks: up each column, then across column totals.
    suffix, ended = tl.associative_scan((x, block_end), 0, _segmented_add, reverse=True)
    column_head, _ = tl.associative_scan(
        (_row(suffix, row, 0), tl.max(block_end.to(tl.int32), axis=0) > 0),
        0,
        _segmented_add,
        reverse=True,
    )
    carry = tl.where(
        lane < LANES - 1, tl.gather(column_head, tl.minimum(lane + 1, LANES - 1), 0), 0.0
    )
    suffix = tl.where(ended, suffix, suffix + carry[None, :])

    # suffix_sum[i - window + 1] for every row i of the region.
    flat = tl.reshape(tl.trans(suffix), (REGION,))
    source = tl.maximum(tl.arange(0, REGION) - halo.to(tl.int32), 0)
    head = tl.trans(tl.reshape(tl.gather(flat, source, 0), (LANES, RUN)))
    # The window starts at a block start iff row i ends a block; before row 0
    # there is no head.
    total = prefix + tl.where(block_end | (g - halo <= 0), 0.0, head)
    # Plain addressing here lets Triton stage the tile through shared memory
    # for coalesced stores.
    t_store = lane[None, :] * RUN + row
    g_store = g0 + t_store
    tl.store(out + g_store, total, mask=(t_store >= halo) & (g_store < n))


TYPES = {"values": "*fp64", "out": "*fp64", "n": "i64", "window": "i64"}
CONSTANTS = {"RUN": 8, "LANES": 256}
NUM_WARPS = 8


def run(values: torch.Tensor, out: torch.Tensor, n: int, window: int) -> None:
    outputs_per_program = CONSTANTS["RUN"] * CONSTANTS["LANES"] - (window - 1)
    grid = (triton.cdiv(n, outputs_per_program),)
    kernel[grid](values, out, n, window, num_warps=NUM_WARPS, **CONSTANTS)
