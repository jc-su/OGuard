import triton
import triton.language as tl


@triton.jit
def kernel(keys, bits, match, n, words, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    key = tl.load(keys + offs, mask=mask, other=0).to(tl.uint64, bitcast=True)
    bit_mask = (tl.cast(words, tl.int64) * 32 - 1).to(tl.uint64)
    p1 = ((key * 0x9E3779B97F4A7C15) >> 32) & bit_mask
    p2 = ((key * 0xC2B2AE3D27D4EB4F) >> 32) & bit_mask
    w1 = tl.load(bits + (p1 >> 5).to(tl.int64), mask=mask, other=0)
    w2 = tl.load(bits + (p2 >> 5).to(tl.int64), mask=mask, other=0)
    hit1 = ((w1 >> (p1 & 31).to(tl.uint32)) & 1) != 0
    hit2 = ((w2 >> (p2 & 31).to(tl.uint32)) & 1) != 0
    tl.store(match + offs, (hit1 & hit2).to(tl.int8), mask=mask)


TYPES = {"keys": "*i64", "bits": "*u32", "match": "*i8", "n": "i64", "words": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(keys, bits, match, n, words):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, bits, match, n, words, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
