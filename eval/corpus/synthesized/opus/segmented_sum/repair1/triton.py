"""Task 12, segmented_sum: sums[j] = sum(values[offsets[j]:offsets[j + 1]]).

The kernel selects one of two mappings from the average segment length
(offsets[g] - offsets[0]) / g. All programs compute the same average, so the
selection is uniform over the grid:

- average >= SPAN_AVERAGE: span mode. Each program sums whole segments, one
  at a time, with a vector of NARROW values per step, or WIDE values per step
  for a segment of WIDE or more values.
- otherwise: row mode. Each program takes a tile of ROWS segments, one per
  lane, and adds one value of every segment of the tile per step. A segment
  of more than LONG values is summed as a span instead, so that one long
  segment does not make the whole tile step through it.

Each program is one warp, so the reductions of a program are warp shuffles.
Each mapping loops over the segments or tiles with a grid stride, so every
grid size gives the correct result; the grid size only changes the speed. An
empty segment gives 0.0. Each segment is summed in a fixed order, so the
result is deterministic.

Every access to values uses an index i with offsets[j] <= i < offsets[j + 1]
for a segment j < g, and the mask of the access compares i with
offsets[j + 1], read from offsets in the same program. In row mode, each lane
holds one segment of the tile, so the mask and the address of an access use
the same row index. With two rows per lane, the compiler computes the second
row index as a bitwise or in the mask and as a sum in the address, which are
equal only because the tile start is a multiple of ROWS. The program reads
the offsets of a long segment again, instead of taking them from the lane of
the segment with a reduction; it masks the lane index from the reduction with
ROWS - 1, which does not change it, and reads and writes only for a segment
below g.

The scalar annotation (tl.int64) sets the Triton type of g at launch. Without
it, Triton types a small Python int as i32. ty does not accept a Triton dtype
in an annotation, and reports each call of a @triton.jit function
(strided_sum, span_sum, tl.zeros, tl.sum, tl.max, tl.min) as an argument
error. The lines where this occurs carry `ty: ignore` comments. A launch takes
host values (tensors, Python scalars, launch options), which the device-side
annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

ROWS = 32  # Segments per tile in row mode: one per lane.
NARROW = 32  # Values per step of a span shorter than WIDE: one per lane.
WIDE = 512  # Values per step of a span of WIDE or more values: 16 per lane.
LONG = 64  # In row mode, a segment of more values is summed as a span.
SPAN_AVERAGE = 32  # The average segment length from which span mode is used.
NUM_WARPS = 1

TYPES = {
    "offsets": "*i32",
    "values": "*fp64",
    "sums": "*fp64",
    "g": "i64",
}
CONSTANTS = {
    "ROWS": ROWS,
    "NARROW": NARROW,
    "WIDE": WIDE,
    "LONG": LONG,
    "SPAN_AVERAGE": SPAN_AVERAGE,
}


@triton.jit
def strided_sum(
    values: tl.tensor, begin: tl.tensor, end: tl.tensor, WIDTH: tl.constexpr
) -> tl.tensor:
    """Return the sum of values[begin:end], WIDTH values per step."""
    acc = tl.zeros([WIDTH], tl.float64)  # ty: ignore[invalid-argument-type]
    for i in tl.range(begin, end, WIDTH):
        index = i + tl.arange(0, WIDTH)
        acc += tl.load(values + index, mask=index < end, other=0.0)
    return tl.sum(acc, 0)  # ty: ignore[invalid-argument-type]


@triton.jit
def span_sum(
    values: tl.tensor,
    begin: tl.tensor,
    end: tl.tensor,
    NARROW: tl.constexpr,
    WIDE: tl.constexpr,
) -> tl.tensor:
    """Return the sum of values[begin:end] with a vector width that fits the length."""
    if end - begin >= WIDE:
        total = strided_sum(values, begin, end, WIDE)  # ty: ignore[invalid-argument-type]
    else:
        total = strided_sum(values, begin, end, NARROW)  # ty: ignore[invalid-argument-type]
    return total


@triton.jit
def kernel(
    offsets: tl.tensor,
    values: tl.tensor,
    sums: tl.tensor,
    g: tl.int64,  # ty: ignore[invalid-type-form]
    ROWS: tl.constexpr,
    NARROW: tl.constexpr,
    WIDE: tl.constexpr,
    LONG: tl.constexpr,
    SPAN_AVERAGE: tl.constexpr,
) -> None:
    program = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    total = tl.load(offsets + g).to(tl.int64) - tl.load(offsets).to(tl.int64)
    if total >= SPAN_AVERAGE * g:
        for j in tl.range(program, g, programs):
            begin = tl.load(offsets + j).to(tl.int64)
            end = tl.load(offsets + j + 1).to(tl.int64)
            tl.store(sums + j, span_sum(values, begin, end, NARROW, WIDE))  # ty: ignore[invalid-argument-type]
    else:
        lane = tl.arange(0, ROWS)
        for base in tl.range(program * ROWS, g, programs * ROWS):
            rows = base + lane
            valid = rows < g
            begin = tl.load(offsets + rows, mask=valid, other=0).to(tl.int64)
            end = tl.load(offsets + rows + 1, mask=valid, other=0).to(tl.int64)
            is_long = (end - begin) > LONG
            short = ~is_long
            # The lane of a short segment reads values[index] for index from
            # begin up to end.
            steps = tl.max(tl.where(short, end - begin, 0), 0)  # ty: ignore[invalid-argument-type]
            acc = tl.zeros([ROWS], tl.float64)  # ty: ignore[invalid-argument-type]
            index = begin
            for _ in tl.range(0, steps):
                acc += tl.load(values + index, mask=short & (index < end), other=0.0)
                index += 1
            tl.store(sums + rows, acc, mask=valid & short)
            # The program sums the long segments of the tile one at a time.
            pending = is_long
            while tl.max(pending.to(tl.int32), 0) > 0:  # ty: ignore[invalid-argument-type]
                row = tl.min(tl.where(pending, lane, ROWS), 0) & (ROWS - 1)  # ty: ignore[invalid-argument-type]
                pending = pending & (lane != row)
                j = base + row
                has = j < g
                span_begin = tl.load(offsets + j, mask=has, other=0).to(tl.int64)
                span_end = tl.load(offsets + j + 1, mask=has, other=0).to(tl.int64)
                span = span_sum(values, span_begin, span_end, NARROW, WIDE)  # ty: ignore[invalid-argument-type]
                tl.store(sums + j, span, mask=has)


def check_array(name: str, array: torch.Tensor, dtype: torch.dtype, length: int) -> None:
    """Raise ValueError unless array is a contiguous CUDA tensor of dtype and shape (length,)."""
    if (
        array.dtype != dtype
        or tuple(array.shape) != (length,)
        or not array.is_cuda
        or not array.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a contiguous CUDA tensor of {dtype} with shape ({length},), "
            f"got {array.dtype} with shape {tuple(array.shape)} on {array.device}"
        )


def programs_for(n: int, g: int) -> int:
    """Return the grid for the mapping that the kernel selects when offsets[0] == 0.

    If offsets[0] > 0, the kernel can select row mode with this grid; the
    grid-stride loops keep that correct.
    """
    if n >= SPAN_AVERAGE * g:
        return g
    return (g + ROWS - 1) // ROWS


def run(
    offsets: torch.Tensor,
    values: torch.Tensor,
    sums: torch.Tensor,
    g: int,
) -> None:
    """Write the sum of each of the g segments into sums."""
    check_array("offsets", offsets, torch.int32, g + 1)
    check_array("values", values, torch.float64, values.shape[0])
    check_array("sums", sums, torch.float64, g)
    if g == 0:
        return
    launch = cast(Callable[..., object], kernel[(programs_for(values.shape[0], g),)])
    launch(offsets, values, sums, g, **CONSTANTS, num_warps=NUM_WARPS)
