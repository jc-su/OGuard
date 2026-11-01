"""Byte-substring search over an Arrow string column (int32 offsets).

Each program owns TILE consecutive strings, one lane per string.

Lane phase: every lane walks its string one byte per step and keeps the last
bytes it read in a shift register, so a step costs one byte load and a few
integer operations whatever the pattern length is. A match ends at byte p when
the register equals the last pattern bytes it holds. The register is one uint32
for k <= 4, two uint32 for k <= 8 and two uint64 for longer patterns (four
compiled variants chosen from k at run time); patterns longer than 16 bytes
verify their leading bytes with extra loads, only in steps where some lane of
the program matched the last 16. The walk stops after LIMIT bytes, so one long
string does not keep all TILE lanes busy.

Long-string phase: each string of the tile longer than LIMIT bytes is then
scanned by all lanes together, CHUNK consecutive match start positions per
step (coalesced loads), for the matches that end at byte LIMIT or later. A
start position is a candidate when the first and the last pattern byte match;
the middle bytes are only checked when the chunk has a candidate, and the scan
of a string stops at its first match.

The empty pattern matches every string.

Bounds: every load and store has a mask that holds 0 <= index and
index < length for the very index it uses, so its bound needs no minimum,
maximum or loop invariant: a string index with n (n + 1 for the end offset),
a pattern index with k, and a byte position with the column length
offsets[n]. The lane phase also stops each lane at offsets[i + 1], the end of
its string i; in the long-string phase the lanes of a chunk read the
consecutive bytes of the chunk inside the column, and only the start
positions that fit in the string can match.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pattern_byte(pattern, k, j):
    """pattern[j] for 0 <= j < k, and 0 elsewhere."""
    return tl.load(pattern + j, mask=(j >= 0) & (j < k), other=0)


@triton.jit
def _target(pattern, k, word, WORD: tl.constexpr):
    """The pattern bytes that register word `word` (0 = newest) holds when a match
    ends at the current byte, packed big-endian, and the mask of those bytes.

    The word holds the pattern positions [hi - size, hi) with hi = k - size * word;
    the positions below 0 are empty. The bytes are loaded as one vector, so every
    position is formed from k and a constant, without a loop over the pattern.
    """
    size: tl.constexpr = WORD.primitive_bitwidth // 8
    lane = tl.arange(0, size)
    position = k - size * word - size + lane
    held = (position >= 0) & (position < k)
    byte = tl.load(pattern + position, mask=held, other=0).to(WORD)
    shift = (8 * (size - 1 - lane)).to(WORD)
    packed = tl.sum(byte << shift, axis=0)
    keep = tl.sum(tl.where(held, tl.full((size,), 255, WORD) << shift, 0).to(WORD), axis=0)
    return packed, keep


@triton.jit
def _scan_lanes(
    chars,
    size,
    start,
    end,
    steps,
    pattern,
    k,
    TILE: tl.constexpr,
    WORD: tl.constexpr,
    TWO: tl.constexpr,
    VERIFY: tl.constexpr,
):
    """Per lane: does the pattern end at some byte p < steps of its string
    [start, end), in a column of `size` bytes."""
    reg_bytes: tl.constexpr = (WORD.primitive_bitwidth // 8) * (2 if TWO else 1)
    want0, keep0 = _target(pattern, k, 0, WORD)
    w0 = tl.zeros((TILE,), WORD)
    if TWO:
        want1, keep1 = _target(pattern, k, 1, WORD)
        w1 = tl.zeros((TILE,), WORD)
    found = tl.zeros((TILE,), tl.int1)
    # The register only needs the last reg_bytes bytes before a match can end.
    for p in range(tl.maximum(k - reg_bytes, 0), steps):
        position = start + p
        active = position < end
        readable = active & (position >= 0) & (position < size)
        if TWO:
            w1 = (w1 << 8) | (w0 >> (WORD.primitive_bitwidth - 8))
        w0 = (w0 << 8) | tl.load(chars + position, mask=readable, other=0).to(WORD)
        hit = active & (p >= k - 1) & ((w0 & keep0) == want0)
        if TWO:
            hit = hit & ((w1 & keep1) == want1)
        if VERIFY:
            if tl.max(hit.to(tl.int32), axis=0) > 0:
                for t in range(0, k - reg_bytes):
                    # A hit has p >= k - 1, so the position is at least start.
                    at = position - k + 1 + t  # below end where hit is set
                    byte = tl.load(chars + at, mask=hit & (at >= 0) & (at < size), other=0)
                    hit = hit & (byte == _pattern_byte(pattern, k, t))
        found = found | hit
    return found


@triton.jit
def kernel(
    offsets,
    chars,
    pattern,
    match,
    n,
    k,
    TILE: tl.constexpr,
    LIMIT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    local = tl.arange(0, TILE)
    s0 = tl.program_id(0).to(tl.int64) * TILE
    s = s0 + local
    in_range = (s >= 0) & (s < n)
    if k == 0:
        tl.store(match + s, tl.full((TILE,), 1, tl.int8), mask=in_range)
    else:
        # The column length offsets[n]; offsets has n + 1 entries.
        size = tl.load(offsets + n, mask=n >= 0, other=0).to(tl.int64)
        start = tl.load(offsets + s, mask=in_range, other=0).to(tl.int64)
        # s + 1 <= n for a string of the column; the other lanes read nothing.
        end = tl.load(offsets + (s + 1), mask=in_range, other=0).to(tl.int64)
        length = end - start
        steps = tl.minimum(tl.max(length, axis=0), LIMIT)
        if k > 16:
            found = _scan_lanes(
                chars, size, start, end, steps, pattern, k, TILE, tl.uint64, True, True
            )
        elif k > 8:
            found = _scan_lanes(
                chars, size, start, end, steps, pattern, k, TILE, tl.uint64, True, False
            )
        elif k > 4:
            found = _scan_lanes(
                chars, size, start, end, steps, pattern, k, TILE, tl.uint32, True, False
            )
        else:
            found = _scan_lanes(
                chars, size, start, end, steps, pattern, k, TILE, tl.uint32, False, False
            )

        first = _pattern_byte(pattern, k, k * 0)
        last = _pattern_byte(pattern, k, k - 1)
        lane = tl.arange(0, CHUNK)
        todo = length > LIMIT
        while tl.max(todo.to(tl.int32), axis=0) > 0:
            pick = tl.min(tl.where(todo, local, TILE), axis=0)
            picked = local == pick
            # The picked string, read again from offsets.
            string = s0 + pick
            present = (string >= 0) & (string < n)
            long_start = tl.load(offsets + string, mask=present, other=0).to(tl.int64)
            long_end = tl.load(offsets + (string + 1), mask=present, other=0).to(tl.int64)
            last_start = long_end - long_start - k
            hits = tl.zeros((), tl.int32)
            # The lane phase covered the matches that end before byte LIMIT.
            q0 = tl.maximum(LIMIT - k + 1, 0)
            while (q0 <= last_start) & (hits == 0):
                q = q0 + lane
                fits = q <= last_start
                # A start that fits keeps the whole match below long_end; the
                # masks below only keep the reads inside the column.
                at_first = long_start + q
                at_last = at_first + k - 1
                byte = tl.load(chars + at_first, mask=(at_first >= 0) & (at_first < size), other=0)
                hit = fits & (byte == first)
                byte = tl.load(chars + at_last, mask=(at_last >= 0) & (at_last < size), other=0)
                hit = hit & (byte == last)
                if tl.max(hit.to(tl.int32), axis=0) > 0:
                    for t in range(1, k - 1):
                        at = at_first + t
                        byte = tl.load(chars + at, mask=(at >= 0) & (at < size), other=0)
                        hit = hit & (byte == _pattern_byte(pattern, k, t))
                    hits = tl.max(hit.to(tl.int32), axis=0)
                q0 = q0 + CHUNK
            found = found | (picked & (hits > 0))
            todo = todo & ~picked
        tl.store(match + s, found.to(tl.int8), mask=in_range)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "pattern": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64",
}
CONSTANTS = {"TILE": 64, "LIMIT": 256, "CHUNK": 256}
NUM_WARPS = 2


def run(
    offsets: torch.Tensor,
    chars: torch.Tensor,
    pattern: torch.Tensor,
    match: torch.Tensor,
    n: int,
    k: int,
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](offsets, chars, pattern, match, n, k, num_warps=NUM_WARPS, **CONSTANTS)
