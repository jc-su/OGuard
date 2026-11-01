import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, bits, match, n, words, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    h1 = (key * -7046029254386353131).to(tl.uint64) >> 32
    h2 = (key * -4417276706812531889).to(tl.uint64) >> 32
    nbits_mask = words * 32 - 1  # words is a power of two
    p1 = h1.to(tl.int64) & nbits_mask
    p2 = h2.to(tl.int64) & nbits_mask
    w1 = tl.load(bits + (p1 >> 5), mask=m, other=0)
    w2 = tl.load(bits + (p2 >> 5), mask=m, other=0)
    b1 = (w1 >> (p1 & 31).to(tl.uint32)) & 1
    b2 = (w2 >> (p2 & 31).to(tl.uint32)) & 1
    tl.store(match + i, (b1 & b2).to(tl.int8), mask=m)


TYPES = {"keys": "*i64", "bits": "*u32", "match": "*i8", "n": "i64", "words": "i64"}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(keys, bits, match, n, words):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        keys, bits, match, n, words, **CONSTANTS, num_warps=NUM_WARPS
    )
