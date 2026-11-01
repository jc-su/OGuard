import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    # Each program hashes BLOCK strings, one lane per string, stepping over
    # the bytes until the longest string of the block is consumed.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    pos = tl.load(offsets + offs, mask=m, other=0).to(tl.int64)
    end = tl.load(offsets + offs + 1, mask=m, other=0).to(tl.int64)
    h = tl.full([BLOCK], 2166136261, tl.uint32)
    while tl.max(end - pos, axis=0) > 0:
        live = pos < end
        c = tl.load(chars + pos, mask=live, other=0).to(tl.uint32)
        h = tl.where(live, (h ^ c) * 16777619, h)
        pos = pos + 1
    tl.store(out + offs, h, mask=m)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(offsets, chars, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
