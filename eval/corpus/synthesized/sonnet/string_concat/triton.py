import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, BLOCK: tl.constexpr):
    # One program per row, copying BLOCK bytes per step: a's bytes, then b's bytes.
    i = tl.program_id(0).to(tl.int64)
    if i < n:
        a_start = tl.load(a_offsets + i).to(tl.int64)
        a_len = tl.load(a_offsets + i + 1).to(tl.int64) - a_start
        b_start = tl.load(b_offsets + i).to(tl.int64)
        b_len = tl.load(b_offsets + i + 1).to(tl.int64) - b_start
        dst = tl.load(out_offsets + i).to(tl.int64)
        for k in range(0, a_len, BLOCK):
            pos = k + tl.arange(0, BLOCK).to(tl.int64)
            mask = pos < a_len
            tl.store(out_chars + dst + pos, tl.load(a_chars + a_start + pos, mask=mask), mask=mask)
        for k in range(0, b_len, BLOCK):
            pos = k + tl.arange(0, BLOCK).to(tl.int64)
            mask = pos < b_len
            tl.store(out_chars + dst + a_len + pos, tl.load(b_chars + b_start + pos, mask=mask), mask=mask)


TYPES = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "out_offsets": "*i32",
    "out_chars": "*u8",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    grid = (max(1, n),)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, **CONSTANTS, num_warps=NUM_WARPS)
