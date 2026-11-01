"""Build a Bloom filter of 32 * words bits from int64 keys.

For each key, set bits h1 % (32 * words) and h2 % (32 * words), where
h1 = (key * 0x9E3779B97F4A7C15) >> 32 and h2 = (key * 0xC2B2AE3D27D4EB4F) >> 32
in wrapping unsigned 64-bit arithmetic. Bit p is bit p % 32 of bits[p / 32].
words is a power of two (1 to 2^26) and bits is zero on entry.

Because 32 * words is a power of two, bit p = h % (32 * words) lies in word
(h >> 5) & (words - 1) at position h & 31, which needs no 32 * words product.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, bits, n, words, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    key = tl.load(keys + offs, mask=in_range, other=0).to(tl.uint64, bitcast=True)
    # The products wrap modulo 2^64; after the shift each hash is below 2^32.
    h1 = ((key * 0x9E3779B97F4A7C15) >> 32).to(tl.int64)
    h2 = ((key * 0xC2B2AE3D27D4EB4F) >> 32).to(tl.int64)
    one = tl.full([BLOCK], 1, tl.uint32)
    tl.atomic_or(
        bits + ((h1 >> 5) & (words - 1)),
        one << (h1 & 31).to(tl.uint32),
        mask=in_range,
        sem="relaxed",
    )
    tl.atomic_or(
        bits + ((h2 >> 5) & (words - 1)),
        one << (h2 & 31).to(tl.uint32),
        mask=in_range,
        sem="relaxed",
    )


TYPES: dict[str, str] = {
    "keys": "*i64",
    "bits": "*u32",
    "n": "i64",
    "words": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(keys: torch.Tensor, bits: torch.Tensor, n: int, words: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, bits, n, words, **CONSTANTS, num_warps=NUM_WARPS)
