import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = rows < n
    start = tl.load(offsets + rows, mask=m, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=m, other=0).to(tl.int64)
    length = end - start
    longest = tl.max(length, axis=0)
    prime = tl.full((BLOCK,), 16777619, tl.uint32)
    h = tl.full((BLOCK,), 2166136261, tl.uint32)
    # FNV-1a is sequential in the bytes of a string, so the strings of the tile
    # advance together and a lane stops updating when its string has ended
    for k in range(0, longest):
        live = k < length
        c = tl.load(chars + start + k, mask=m & live, other=0).to(tl.uint32)
        h = tl.where(live, (h ^ c) * prime, h)
    tl.store(out + rows, h, mask=m)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(offsets, chars, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, out, n, **CONSTANTS, num_warps=NUM_WARPS)
