"""string_hash: out[i] = 32-bit FNV-1a hash of the bytes of string i.

FNV-1a is sequential within a string, so each lane hashes one string; the program
steps through byte positions up to the longest string of its block, with lanes
past the end of their string masked.
"""

import triton
import triton.language as tl

BLOCK = 256
NUM_WARPS = 4

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    begin = tl.load(offsets + rows, mask=in_range, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=in_range, other=0).to(tl.int64)
    length = end - begin
    h = tl.full((BLOCK,), 2166136261, tl.uint32)
    for k in range(0, tl.max(length, axis=0)):
        active = k < length
        c = tl.load(chars + begin + k, mask=active, other=0).to(tl.uint32)
        h = tl.where(active, (h ^ c) * 16777619, h)
    tl.store(out + rows, h, mask=in_range)


def run(offsets, chars, out, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](offsets, chars, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
