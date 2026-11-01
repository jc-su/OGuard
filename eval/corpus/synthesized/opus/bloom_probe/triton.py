"""Probe a Bloom filter of 32 * words bits with int64 keys.

match[i] = 1 when bits h1 % (32 * words) and h2 % (32 * words) of the filter
are both set, where h1 = (key * 0x9E3779B97F4A7C15) >> 32 and
h2 = (key * 0xC2B2AE3D27D4EB4F) >> 32 in wrapping unsigned 64-bit arithmetic,
and 0 otherwise. Bit p is bit p % 32 of bits[p / 32]; words is a power of two
(1 to 2^26), so bit p = h % (32 * words) lies in word (h >> 5) & (words - 1)
at position h & 31.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, bits, match, n, words, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    key = tl.load(keys + offs, mask=in_range, other=0).to(tl.uint64, bitcast=True)
    # The products wrap modulo 2^64; after the shift each hash is below 2^32.
    h1 = ((key * 0x9E3779B97F4A7C15) >> 32).to(tl.int64)
    h2 = ((key * 0xC2B2AE3D27D4EB4F) >> 32).to(tl.int64)
    word1 = tl.load(bits + ((h1 >> 5) & (words - 1)), mask=in_range, other=0)
    hit1 = ((word1 >> (h1 & 31).to(tl.uint32)) & 1) != 0
    # The second word is read only for the keys whose first bit is set.
    word2 = tl.load(bits + ((h2 >> 5) & (words - 1)), mask=in_range & hit1, other=0)
    hit2 = ((word2 >> (h2 & 31).to(tl.uint32)) & 1) != 0
    tl.store(match + offs, (hit1 & hit2).to(tl.int8), mask=in_range)


TYPES: dict[str, str] = {
    "keys": "*i64",
    "bits": "*u32",
    "match": "*i8",
    "n": "i64",
    "words": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(keys: torch.Tensor, bits: torch.Tensor, match: torch.Tensor, n: int, words: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, bits, match, n, words, **CONSTANTS, num_warps=NUM_WARPS)
