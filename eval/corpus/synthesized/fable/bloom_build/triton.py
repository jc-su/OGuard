import triton
import triton.language as tl


@triton.jit
def kernel(keys, bits, n, words, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    key = tl.load(keys + offs, mask=mask, other=0).to(tl.uint64, bitcast=True)
    bit_mask = (tl.cast(words, tl.int64) * 32 - 1).to(tl.uint64)
    p1 = ((key * 0x9E3779B97F4A7C15) >> 32) & bit_mask
    p2 = ((key * 0xC2B2AE3D27D4EB4F) >> 32) & bit_mask
    one = tl.full([BLOCK], 1, tl.uint32)
    tl.atomic_or(
        bits + (p1 >> 5).to(tl.int64), one << (p1 & 31).to(tl.uint32), mask=mask, sem="relaxed"
    )
    tl.atomic_or(
        bits + (p2 >> 5).to(tl.int64), one << (p2 & 31).to(tl.uint32), mask=mask, sem="relaxed"
    )


TYPES = {"keys": "*i64", "bits": "*u32", "n": "i64", "words": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(keys, bits, n, words):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, bits, n, words, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
