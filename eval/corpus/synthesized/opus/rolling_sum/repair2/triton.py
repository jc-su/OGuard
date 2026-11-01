"""Trailing rolling sum over a fixed window of 1 to 1024 rows.

Van Herk / Gil-Werman scheme: cut the rows into segments of `window` rows
aligned at multiples of `window`. The window [i - window + 1, i] then covers
the suffix of one segment and the prefix of the next, so
    out[i] = suffix_sum[i - window + 1] + prefix_sum[i],
where both sums run inside a segment; when i - window + 1 starts a segment the
window is prefix_sum[i] alone. Every output is a sum of at most `window` inputs
and no partial sum is subtracted, so the rounding error is that of summing the
window directly.

Each program writes the outputs [t0, t0 + count), where count is a multiple of
the window (so t0 starts a segment) and count <= REGION - window + 2. It holds
REGION positions k as a [RUN, LANES] tile whose column l is the run of
positions l * RUN .. l * RUN + RUN - 1 and lives in one thread, and loads two
views of the column in the same positions: x[k] = values[t0 + k] and
y[k] = values[t0 + k - window + 1]. A segmented scan of x from the left gives
prefix_sum[t0 + k] at k, because every segment of an output starts at t0 or
later, and a segmented scan of y from the right gives
suffix_sum[t0 + k - window + 1] at k, because that segment ends at or before
k + window - 2 < REGION. So the two sums of an output are at the same position.

The RTX A6000 (GA102) has two float64 units per SM and float64 additions bound
this kernel, so each scan runs sequentially down the columns and then once
across the LANES column results. The carry into a column is the exclusive
scan of the column results, which a scan of (inclusive, exclusive) pairs gives
without moving values between lanes outside the scans.

Bounds: every load and store has the mask 0 <= row and row < n of the very
row it uses, so its bound needs no minimum, maximum or loop invariant; a row
before 0 or past n - 1 reads as 0. An output row is t0 + k, stored when k is
below the program's output count and the row is in [0, n).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _segmented_add(value_a, flag_a, value_b, flag_b):
    # flag_b marks a segment boundary at b: the running sum restarts there.
    return tl.where(flag_b, value_b, value_a + value_b), flag_a | flag_b


@triton.jit
def _segmented_add_exclusive(inc_a, flag_a, exc_a, flag_exc_a, inc_b, flag_b, exc_b, flag_exc_b):
    # (inc, flag) is the segmented sum of a range, and (exc, flag_exc) the same
    # sum without the range's last element in scan order; a single element has
    # exc = 0 and flag_exc = False, so a scan leaves the exclusive sum in exc.
    inc = tl.where(flag_b, inc_b, inc_a + inc_b)
    exc = tl.where(flag_exc_b, exc_b, inc_a + exc_b)
    return inc, flag_a | flag_b, exc, flag_a | flag_exc_b


@triton.jit
def _row(x, row, r):
    """Row r of a [RUN, LANES] float64 tile, selected without float64 additions."""
    bits = tl.where(row == r, x.to(tl.int64, bitcast=True), 0)
    return tl.sum(bits, axis=0).to(tl.float64, bitcast=True)


@triton.jit
def _load(values, n, first, t):
    """values[first + t], and 0 for rows outside [0, n)."""
    rows = first + t
    return tl.load(values + rows, mask=(rows >= 0) & (rows < n), other=0.0)


@triton.jit
def _segmented_scan(x, flag, row, REVERSE: tl.constexpr, RUN: tl.constexpr):
    """Segmented scan of a [RUN, LANES] tile in the order of its positions.

    The scan runs down each column, then across the column results: the carry
    into a column is the exclusive scan of the columns' last results in scan
    order.
    """
    local, seen = tl.associative_scan((x, flag), 0, _segmented_add, reverse=REVERSE)
    last = _row(local, row, 0 if REVERSE else RUN - 1)
    has_flag = tl.max(flag.to(tl.int32), axis=0) > 0
    _, _, carry, _ = tl.associative_scan(
        (last, has_flag, tl.zeros_like(last), tl.zeros_like(has_flag)),
        0,
        _segmented_add_exclusive,
        reverse=REVERSE,
    )
    return tl.where(seen, local, local + carry[None, :])


@triton.jit
def kernel(values, out, n, window, RUN: tl.constexpr, LANES: tl.constexpr):
    REGION: tl.constexpr = RUN * LANES
    w = tl.cast(window, tl.int64)
    w32 = w.to(tl.int32)
    # Outputs per program: a multiple of the window, at most REGION - w + 2 and REGION.
    count = tl.minimum((REGION - w + 2) // w * w, REGION)
    t0 = tl.program_id(0).to(tl.int64) * count
    row = tl.arange(0, RUN)[:, None]
    lane = tl.arange(0, LANES)
    # Multiplying by a run-time 1 hides the contiguity of a column from Triton's
    # coalescing pass, which then keeps each column in one thread.
    runtime_one = (w32 // w32).to(tl.int32)
    k = lane[None, :] * RUN + row * runtime_one

    # Prefix sums of x: a segment starts at k when k % w == 0.
    x = _load(values, n, t0, k)
    prefix = _segmented_scan(x, k % w32 == 0, row, False, RUN)
    # Suffix sums of y: a segment ends at k when (k + 2) % w == 0.
    y = _load(values, n, t0 - w + 1, k)
    suffix = _segmented_scan(y, (k + 2) % w32 == 0, row, True, RUN)

    # The window of output t0 + k starts a segment when (k + 1) % w == 0. The
    # store keeps the column layout, so the tile does not go through shared
    # memory. Positions past the program's outputs are not stored.
    total = prefix + tl.where((k + 1) % w32 == 0, 0.0, suffix)
    out_row = t0 + k
    tl.store(out + out_row, total, mask=(k < count) & (out_row >= 0) & (out_row < n))


TYPES = {"values": "*fp64", "out": "*fp64", "n": "i64", "window": "i64"}
CONSTANTS = {"RUN": 8, "LANES": 256}
NUM_WARPS = 8


def run(values: torch.Tensor, out: torch.Tensor, n: int, window: int) -> None:
    region = CONSTANTS["RUN"] * CONSTANTS["LANES"]
    outputs_per_program = min((region - window + 2) // window * window, region)
    grid = (triton.cdiv(n, outputs_per_program),)
    kernel[grid](values, out, n, window, num_warps=NUM_WARPS, **CONSTANTS)
