"""string_length: number of UTF-8 code points per string.

A code point starts at every byte that is not a continuation byte
((byte & 0xC0) != 0x80), so the length is the count of such bytes. The kernel
picks one of two paths per launch from the average string length:

- short strings (average below WARP_MIN_AVG bytes): a program handles BLOCK_S
  strings; one round over a [BLOCK_S, BYTES_A] tile reads BYTES_A bytes of
  every string, and the rounds run up to the longest string of the tile;
- long strings: programs walk the strings STRINGS_B at a time with a grid
  stride and a [STRINGS_B, BYTES_B] tile, so each round reads BYTES_B
  consecutive bytes of a string with coalesced loads.

Both paths accumulate the start flags in the tile and reduce once per string.
The kernel picks the path from offsets[n]; run picks the grid for that path
from the length of chars, which the host knows without a device sync. Both
paths are correct on any grid, so a mismatch only costs speed.

Byte positions are the 32-bit sum offsets[s] + j taken as unsigned, and a
byte is read only when that position is below offsets[s + 1], compared as
unsigned. Under the contract (nondecreasing offsets with offsets[n] = length
of chars) this is exactly j < offsets[s + 1] - offsets[s]: a sum that wraps
past INT_MAX compares above offsets[s + 1] and is skipped, and such a j is
past the end of the string. The position is zero-extended into the address,
so it is non-negative by construction and below offsets[s + 1], the length
of chars at most, on every load.
"""

import triton
import triton.language as tl

BLOCK_S = 256
BYTES_A = 4
STRINGS_B = 64
BYTES_B = 32
WARP_MIN_AVG = 64
NUM_WARPS = 4

TYPES = {"offsets": "*i32", "chars": "*u8", "lengths": "*i32", "n": "i64"}
CONSTANTS = {
    "BLOCK_S": BLOCK_S,
    "BYTES_A": BYTES_A,
    "STRINGS_B": STRINGS_B,
    "BYTES_B": BYTES_B,
    "WARP_MIN_AVG": WARP_MIN_AVG,
}


@triton.jit
def count_starts(offsets, chars, lengths, s, valid, S: tl.constexpr, BYTES: tl.constexpr):
    begin = tl.load(offsets + s, mask=valid, other=0)
    end = tl.load(offsets + s + 1, mask=valid, other=0)
    longest = tl.max(end - begin, axis=0)
    end_u = end.to(tl.uint32)[:, None]
    acc = tl.zeros([S, BYTES], tl.int32)
    for j0 in range(0, longest, BYTES):
        j = j0 + tl.arange(0, BYTES)
        p = (begin[:, None] + j[None, :]).to(tl.uint32)
        in_string = p < end_u
        # 0x80 is a continuation byte, so padding is never counted.
        c = tl.load(chars + p.to(tl.int64), mask=in_string, other=0x80)
        acc += ((c & 0xC0) != 0x80).to(tl.int32)
    tl.store(lengths + s, tl.sum(acc, axis=1), mask=valid)


@triton.jit
def kernel(
    offsets,
    chars,
    lengths,
    n: tl.int64,
    BLOCK_S: tl.constexpr,
    BYTES_A: tl.constexpr,
    STRINGS_B: tl.constexpr,
    BYTES_B: tl.constexpr,
    WARP_MIN_AVG: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    total = tl.load(offsets + n)
    if total >= n * WARP_MIN_AVG:
        stride = tl.num_programs(0).to(tl.int64) * STRINGS_B
        for s0 in range(pid * STRINGS_B, n, stride):
            s = s0 + tl.arange(0, STRINGS_B)
            count_starts(offsets, chars, lengths, s, s < n, STRINGS_B, BYTES_B)
    else:
        s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
        count_starts(offsets, chars, lengths, s, s < n, BLOCK_S, BYTES_A)


def run(offsets, chars, lengths, n):
    wide = chars.numel() >= n * WARP_MIN_AVG
    grid = (max(1, triton.cdiv(n, STRINGS_B if wide else BLOCK_S)),)
    kernel[grid](offsets, chars, lengths, n, **CONSTANTS, num_warps=NUM_WARPS)
