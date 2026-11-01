"""starts_with: match[i] = 1 when string i begins with the k-byte pattern prefix.

A program handles a tile of BLOCK strings. A string shorter than k never
matches; the others compare their first k bytes with the pattern KT bytes at a
time through a [BLOCK, KT] tile, so a short pattern costs one round of loads
that fall in the first sector of each string.

Byte positions are formed in 64-bit from the string start, and every load of
chars is masked by 0 <= position < offsets[n], the length of chars, as well
as by the string and pattern bounds. Under the contract (nondecreasing
offsets with offsets[n] = length of chars) the length mask always holds for
a string at least k bytes long, so it changes no result.
"""

import triton
import triton.language as tl

BLOCK = 256
KT = 4
NUM_WARPS = 4

TYPES = {"offsets": "*i32", "chars": "*u8", "prefix": "*u8", "match": "*i8", "n": "i64", "k": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "KT": KT}


@triton.jit
def kernel(
    offsets, chars, prefix, match, n: tl.int64, k: tl.int64, BLOCK: tl.constexpr, KT: tl.constexpr
):
    pid = tl.program_id(0)
    total = tl.load(offsets + n)
    s = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = s < n
    begin = tl.load(offsets + s, mask=valid, other=0)
    end = tl.load(offsets + s + 1, mask=valid, other=0)
    ok = valid & ((end - begin) >= k)
    for j0 in range(0, k, KT):
        j = j0 + tl.arange(0, KT)
        in_pattern = j < k
        pat = tl.load(prefix + j, mask=in_pattern, other=0)
        p = begin.to(tl.int64)[:, None] + j[None, :]
        in_chars = ok[:, None] & in_pattern[None, :] & (p >= 0) & (p < total)
        c = tl.load(chars + p, mask=in_chars, other=0)
        # Padded pattern bytes and padded string bytes are both 0, so they compare equal.
        equal = (c == pat[None, :]).to(tl.int32)
        ok = ok & (tl.sum(equal, axis=1) == KT)
    tl.store(match + s, ok.to(tl.int8), mask=valid)


def run(offsets, chars, prefix, match, n, k):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](offsets, chars, prefix, match, n, k, **CONSTANTS, num_warps=NUM_WARPS)
