import triton
import triton.language as tl


@triton.jit
def kernel(packed, out, n, w, bits, BLOCK: tl.constexpr):
    # Each program decodes BLOCK codes: code i starts at bit i * bits of the
    # stream and spans at most two words.
    pid = tl.program_id(0).to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = i < n
    p = i * bits
    word = p >> 5
    sh = p & 31
    lo = tl.load(packed + word, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.uint64)
    need_hi = m & (sh + bits > 32)
    hi = tl.load(packed + word + 1, mask=need_hi, other=0).to(tl.uint32, bitcast=True).to(tl.uint64)
    acc = (lo >> sh.to(tl.uint64)) | (hi << (32 - sh).to(tl.uint64))
    keep = (tl.full([BLOCK], 1, tl.uint64) << (i * 0 + bits).to(tl.uint64)) - 1
    tl.store(out + i, (acc & keep).to(tl.uint32), mask=m)


TYPES = {"packed": "*u32", "out": "*u32", "n": "i64", "w": "i64", "bits": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(packed, out, n, w, bits):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](packed, out, n, w, bits, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
