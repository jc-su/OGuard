"""Task 9, string_length: the number of UTF-8 code points of each string.

The string column is in the Arrow layout: string i is the byte range
chars[offsets[i]:offsets[i + 1]]. A code point starts at every byte that is
not a continuation byte (0b10xxxxxx), so the kernel counts those bytes.

The kernel selects one of two mappings from the average string length in
bytes, (offsets[n] - offsets[0]) / n. All programs compute the same average,
so the selection is uniform over the grid:

- average >= SPAN_AVERAGE: span mode. Each program counts whole strings, one
  at a time, with a vector of NARROW bytes per step, or WIDE bytes per step
  for a string of WIDE or more bytes.
- otherwise: row mode. Each program takes a tile of ROWS strings, one per
  thread, and reads one byte of every string of the tile per step. A string
  of more than LONG bytes is counted as a span instead, so that one long
  string does not make the whole tile step through it.

Each program has NUM_WARPS warps, so ROWS and NARROW are one value per
thread, and a reduction over a program combines the warps through shared
memory. Each mapping loops over the strings or tiles with a grid stride, so
every grid size gives the correct result; the grid size only changes the
speed.

Every access to chars uses an index j with offsets[i] <= j < offsets[i + 1]
for a string i < n, and the mask of the access compares j with
offsets[i + 1], read from offsets in the same program. In row mode, each
thread holds one string of the tile, so the mask and the address of an access
use the same row index. With two rows per thread, the compiler computes the
second row index as a bitwise or in the mask and as a sum in the address,
which are equal only because the tile start is a multiple of ROWS. The
program reads the offsets of a long string again, instead of taking them from
the thread of the string with a reduction; it masks the row index from the
reduction with ROWS - 1, which does not change it, and reads and writes only
for a string below n.

In row mode, the step count of the short strings of a tile is the maximum of
their lengths over the threads, so a thread cannot bound it from its own
values. The kernel takes the minimum of the count and LONG, which does not
change it, because a short string has at most LONG bytes. The index of a
thread then advances at most LONG times from offsets[i], so it cannot wrap
around before the mask compares it with offsets[i + 1].

The scalar annotation (tl.int64) sets the Triton type of n at launch. Without
it, Triton types a small Python int as i32. ty does not accept a Triton dtype
in an annotation, and reports each call of a @triton.jit function
(strided_count, span_count, tl.zeros, tl.sum, tl.max, tl.min) as an argument
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

NUM_WARPS = 2
ROWS = 32 * NUM_WARPS  # Strings per tile in row mode: one per thread.
NARROW = 32 * NUM_WARPS  # Bytes per step of a span shorter than WIDE: one per thread.
WIDE = 512 * NUM_WARPS  # Bytes per step of a span of WIDE or more bytes: 16 per thread.
LONG = 256  # In row mode, a string of more bytes is counted as a span.
SPAN_AVERAGE = 256  # The average string length in bytes from which span mode is used.

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "lengths": "*i32",
    "n": "i64",
}
CONSTANTS = {
    "ROWS": ROWS,
    "NARROW": NARROW,
    "WIDE": WIDE,
    "LONG": LONG,
    "SPAN_AVERAGE": SPAN_AVERAGE,
}


@triton.jit
def is_lead(byte: tl.tensor) -> tl.tensor:
    """Return 1 for each byte that starts a code point, as int32, and 0 for a continuation byte."""
    return ((byte & 0xC0) != 0x80).to(tl.int32)


@triton.jit
def strided_count(
    chars: tl.tensor, begin: tl.tensor, end: tl.tensor, WIDTH: tl.constexpr
) -> tl.tensor:
    """Return the number of code points in chars[begin:end], WIDTH bytes per step."""
    acc = tl.zeros([WIDTH], tl.int32)  # ty: ignore[invalid-argument-type]
    for i in tl.range(begin, end, WIDTH):
        index = i + tl.arange(0, WIDTH)
        # A masked read gives a continuation byte, which adds 0.
        acc += is_lead(tl.load(chars + index, mask=index < end, other=0x80))  # ty: ignore[invalid-argument-type]
    return tl.sum(acc, 0)  # ty: ignore[invalid-argument-type]


@triton.jit
def span_count(
    chars: tl.tensor,
    begin: tl.tensor,
    end: tl.tensor,
    NARROW: tl.constexpr,
    WIDE: tl.constexpr,
) -> tl.tensor:
    """Return the number of code points in chars[begin:end] with a vector width that fits the length."""
    if end - begin >= WIDE:
        count = strided_count(chars, begin, end, WIDE)  # ty: ignore[invalid-argument-type]
    else:
        count = strided_count(chars, begin, end, NARROW)  # ty: ignore[invalid-argument-type]
    return count


@triton.jit
def kernel(
    offsets: tl.tensor,
    chars: tl.tensor,
    lengths: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    ROWS: tl.constexpr,
    NARROW: tl.constexpr,
    WIDE: tl.constexpr,
    LONG: tl.constexpr,
    SPAN_AVERAGE: tl.constexpr,
) -> None:
    program = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    total = tl.load(offsets + n).to(tl.int64) - tl.load(offsets).to(tl.int64)
    if total >= SPAN_AVERAGE * n:
        for i in tl.range(program, n, programs):
            begin = tl.load(offsets + i).to(tl.int64)
            end = tl.load(offsets + i + 1).to(tl.int64)
            tl.store(lengths + i, span_count(chars, begin, end, NARROW, WIDE))  # ty: ignore[invalid-argument-type]
    else:
        tile_row = tl.arange(0, ROWS)
        for base in tl.range(program * ROWS, n, programs * ROWS):
            rows = base + tile_row
            valid = rows < n
            begin = tl.load(offsets + rows, mask=valid, other=0).to(tl.int64)
            end = tl.load(offsets + rows + 1, mask=valid, other=0).to(tl.int64)
            is_long = (end - begin) > LONG
            short = ~is_long
            # The thread of a short string reads chars[index] for index from
            # begin up to end. A masked read gives a continuation byte, which
            # adds 0. A short string has at most LONG bytes, so the minimum
            # with LONG does not change the step count.
            steps = tl.minimum(tl.max(tl.where(short, end - begin, 0), 0), LONG)  # ty: ignore[invalid-argument-type]
            acc = tl.zeros([ROWS], tl.int32)  # ty: ignore[invalid-argument-type]
            index = begin
            for _ in tl.range(0, steps):
                byte = tl.load(chars + index, mask=short & (index < end), other=0x80)
                acc += is_lead(byte)  # ty: ignore[invalid-argument-type]
                index += 1
            tl.store(lengths + rows, acc, mask=valid & short)
            # The program counts the long strings of the tile one at a time.
            pending = is_long
            while tl.max(pending.to(tl.int32), 0) > 0:  # ty: ignore[invalid-argument-type]
                row = tl.min(tl.where(pending, tile_row, ROWS), 0) & (ROWS - 1)  # ty: ignore[invalid-argument-type]
                pending = pending & (tile_row != row)
                i = base + row
                has = i < n
                span_begin = tl.load(offsets + i, mask=has, other=0).to(tl.int64)
                span_end = tl.load(offsets + i + 1, mask=has, other=0).to(tl.int64)
                span = span_count(chars, span_begin, span_end, NARROW, WIDE)  # ty: ignore[invalid-argument-type]
                tl.store(lengths + i, span, mask=has)


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


def programs_for(total: int, n: int) -> int:
    """Return the grid for the mapping that the kernel selects when offsets[0] == 0.

    total is the byte count of chars. If offsets[0] > 0, the kernel can
    select row mode with this grid; the grid-stride loops keep that correct.
    """
    if total >= SPAN_AVERAGE * n:
        return n
    return (n + ROWS - 1) // ROWS


def run(
    offsets: torch.Tensor,
    chars: torch.Tensor,
    lengths: torch.Tensor,
    n: int,
) -> None:
    """Write the code point count of each of the n strings into lengths."""
    check_array("offsets", offsets, torch.int32, n + 1)
    check_array("chars", chars, torch.uint8, chars.shape[0])
    check_array("lengths", lengths, torch.int32, n)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[(programs_for(chars.shape[0], n),)])
    launch(offsets, chars, lengths, n, **CONSTANTS, num_warps=NUM_WARPS)
