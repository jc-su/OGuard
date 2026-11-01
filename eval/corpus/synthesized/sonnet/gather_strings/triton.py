import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, idx, out_offsets, out_chars, n, m, BLOCK: tl.constexpr):
    # One program per output string, copying BLOCK bytes per step.
    i = tl.program_id(0).to(tl.int64)
    if i < m:
        src = tl.load(idx + i).to(tl.int64)
        start = tl.load(offsets + src).to(tl.int64)
        length = tl.load(offsets + src + 1).to(tl.int64) - start
        dst = tl.load(out_offsets + i).to(tl.int64)
        for k in range(0, length, BLOCK):
            pos = k + tl.arange(0, BLOCK).to(tl.int64)
            mask = pos < length
            tl.store(out_chars + dst + pos, tl.load(chars + start + pos, mask=mask), mask=mask)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "idx": "*i32",
    "out_offsets": "*i32",
    "out_chars": "*u8",
    "n": "i64",
    "m": "i64",
}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    grid = (max(1, m),)
    kernel[grid](offsets, chars, idx, out_offsets, out_chars, n, m, **CONSTANTS, num_warps=NUM_WARPS)
