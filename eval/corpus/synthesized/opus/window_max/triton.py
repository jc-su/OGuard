"""Sliding window maximum over an int64 column, in Triton."""

import torch
import triton
import triton.language as tl

LOWEST: tl.constexpr = tl.constexpr(-(2**63))


@triton.jit
def _segmented_max(acc, acc_starts, x, x_starts):
    # Combine step of a segmented inclusive max scan: an element that starts a
    # segment in the scan direction drops the accumulated maximum.
    return tl.where(x_starts, x, tl.maximum(acc, x)), acc_starts | x_starts


@triton.jit(do_not_specialize=["n", "w"])
def kernel(values, out, n, w, BLOCK: tl.constexpr, HALO: tl.constexpr):
    # Van Herk / Gil-Werman algorithm. Split the positions into segments of w
    # rows that start at multiples of w. For row i and j = i - w + 1, the window
    # [j, i] is the end of the segment of j followed by the start of the segment
    # of i (the same segment when j is a multiple of w), so
    #   out[i] = max(suffix max of j in its segment, prefix max of i in its segment).
    # Each program computes BLOCK outputs. Both scans run on lanes indexed by the
    # output row. The segments cut by the tile edges extend at most w - 1 rows
    # outside the tile, which a reduction over HALO lanes covers since w <= HALO.
    tl.static_assert(BLOCK >= HALO)
    width = w.to(tl.int64)
    width32 = w.to(tl.int32)
    first = tl.program_id(0).to(tl.int64) * BLOCK
    lane = tl.arange(0, BLOCK)
    halo = tl.arange(0, HALO)

    # Prefix maxima of rows i in their segments.
    i = first + lane
    x = tl.load(values + i, mask=i < n, other=LOWEST)
    i_phase = ((first % width).to(tl.int32) + lane) % width32
    prefix, _ = tl.associative_scan((x, i_phase == 0), 0, _segmented_max)
    # The rows of the first segment that come before the tile.
    first_begin = first - first % width
    before = first_begin + halo
    carry_in = tl.max(tl.load(values + before, mask=before < first, other=LOWEST), axis=0)
    prefix = tl.where(i < first_begin + width, tl.maximum(prefix, carry_in), prefix)

    # Suffix maxima of rows j = i - w + 1 in their segments. Rows j < 0 belong
    # to the segment [-w, -1] and hold the identity.
    j = i - width + 1
    y = tl.load(values + j, mask=(j >= 0) & (j < n), other=LOWEST)
    j_phase = (((first + 1) % width).to(tl.int32) + lane) % width32
    suffix, _ = tl.associative_scan((y, j_phase == width32 - 1), 0, _segmented_max, reverse=True)
    # The rows of the last segment that come after the tile. Since
    # w <= HALO <= BLOCK, the first of them is at least BLOCK - w + 1 >= 1.
    last = first + BLOCK - width
    last_end = last + width - 1 - (first + BLOCK) % width
    after = last + 1 + halo
    carry_out = tl.max(tl.load(values + after, mask=(after <= last_end) & (after < n), other=LOWEST), axis=0)
    suffix = tl.where(j > last_end - width, tl.maximum(suffix, carry_out), suffix)

    tl.store(out + i, tl.maximum(prefix, suffix), mask=i < n)


TYPES: dict[str, str] = {"values": "*i64", "out": "*i64", "n": "i64", "w": "i64"}
CONSTANTS: dict[str, int] = {"BLOCK": 1024, "HALO": 1024}
NUM_WARPS: int = 4


def run(values: torch.Tensor, out: torch.Tensor, n: int, w: int) -> None:
    if not 1 <= w <= CONSTANTS["HALO"]:
        raise ValueError(f"w must be in [1, {CONSTANTS['HALO']}], got {w}")
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, out, n, w, **CONSTANTS, num_warps=NUM_WARPS)
