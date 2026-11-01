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
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _target(pattern, k, word, WORD: tl.constexpr):
    """The pattern bytes that register word `word` (0 = newest) holds when a match
    ends at the current byte, packed big-endian, and the mask of those bytes."""
    size: tl.constexpr = WORD.primitive_bitwidth // 8
    hi = tl.maximum(k - size * word, 0)
    lo = tl.maximum(hi - size, 0)
    packed = tl.zeros((), WORD)
    for j in range(lo, hi):
        packed = (packed << 8) | tl.load(pattern + j).to(WORD)
    ones = tl.zeros((), WORD) - 1
    bits = 8 * (hi - lo)
    keep = tl.where(bits > 0, ones >> (WORD.primitive_bitwidth - tl.maximum(bits, 1)).to(WORD), 0)
    return packed, keep


@triton.jit
def _scan_lanes(
    text,
    length,
    steps,
    pattern,
    k,
    TILE: tl.constexpr,
    WORD: tl.constexpr,
    TWO: tl.constexpr,
    VERIFY: tl.constexpr,
):
    """Per lane: does the pattern end at some byte p < steps of its string."""
    reg_bytes: tl.constexpr = (WORD.primitive_bitwidth // 8) * (2 if TWO else 1)
    want0, keep0 = _target(pattern, k, 0, WORD)
    w0 = tl.zeros((TILE,), WORD)
    if TWO:
        want1, keep1 = _target(pattern, k, 1, WORD)
        w1 = tl.zeros((TILE,), WORD)
    found = tl.zeros((TILE,), tl.int1)
    # The register only needs the last reg_bytes bytes before a match can end.
    for p in range(tl.maximum(k - reg_bytes, 0), steps):
        active = p < length
        if TWO:
            w1 = (w1 << 8) | (w0 >> (WORD.primitive_bitwidth - 8))
        w0 = (w0 << 8) | tl.load(text + p, mask=active, other=0).to(WORD)
        hit = active & (p >= k - 1) & ((w0 & keep0) == want0)
        if TWO:
            hit = hit & ((w1 & keep1) == want1)
        if VERIFY:
            if tl.max(hit.to(tl.int32), axis=0) > 0:
                for t in range(0, k - reg_bytes):
                    byte = tl.load(text + (p - k + 1 + t), mask=hit, other=0)
                    hit = hit & (byte == tl.load(pattern + t))
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
    s = tl.program_id(0).to(tl.int64) * TILE + local
    in_range = s < n
    if k == 0:
        tl.store(match + s, tl.full((TILE,), 1, tl.int8), mask=in_range)
    else:
        start = tl.load(offsets + s, mask=in_range, other=0)
        length = tl.load(offsets + s + 1, mask=in_range, other=0) - start
        text = chars + start
        steps = tl.minimum(tl.max(length, axis=0), LIMIT)
        if k > 16:
            found = _scan_lanes(text, length, steps, pattern, k, TILE, tl.uint64, True, True)
        elif k > 8:
            found = _scan_lanes(text, length, steps, pattern, k, TILE, tl.uint64, True, False)
        elif k > 4:
            found = _scan_lanes(text, length, steps, pattern, k, TILE, tl.uint32, True, False)
        else:
            found = _scan_lanes(text, length, steps, pattern, k, TILE, tl.uint32, False, False)

        first = tl.load(pattern)
        last = tl.load(pattern + k - 1)
        lane = tl.arange(0, CHUNK)
        todo = length > LIMIT
        while tl.max(todo.to(tl.int32), axis=0) > 0:
            pick = tl.min(tl.where(todo, local, TILE), axis=0)
            picked = local == pick
            long_text = chars + tl.sum(tl.where(picked, start, 0), axis=0)
            last_start = tl.sum(tl.where(picked, length, 0), axis=0) - k
            hits = tl.zeros((), tl.int32)
            # The lane phase covered the matches that end before byte LIMIT.
            q0 = tl.maximum(LIMIT - k + 1, 0)
            while (q0 <= last_start) & (hits == 0):
                q = q0 + lane
                fits = q <= last_start
                hit = fits & (tl.load(long_text + q, mask=fits, other=0) == first)
                hit = hit & (tl.load(long_text + q + k - 1, mask=fits, other=0) == last)
                if tl.max(hit.to(tl.int32), axis=0) > 0:
                    for t in range(1, k - 1):
                        byte = tl.load(long_text + q + t, mask=hit, other=0)
                        hit = hit & (byte == tl.load(pattern + t))
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
