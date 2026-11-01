import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a_offsets, a_chars, b_offsets, b_chars, match, n, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    a0 = tl.load(a_offsets + r).to(tl.int64)
    la = tl.load(a_offsets + r + 1).to(tl.int64) - a0
    b0 = tl.load(b_offsets + r).to(tl.int64)
    lb = tl.load(b_offsets + r + 1).to(tl.int64) - b0
    same = la == lb
    limit = tl.where(same, la, 0)
    diff = tl.zeros((BLOCK,), dtype=tl.int32)
    for off in range(0, limit, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < limit
        x = tl.load(a_chars + a0 + k, mask=m, other=0)
        y = tl.load(b_chars + b0 + k, mask=m, other=0)
        diff += (x != y).to(tl.int32)
    eq = same & (tl.sum(diff, axis=0) == 0)
    tl.store(match + r, eq.to(tl.int8))


TYPES = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "match": "*i8",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 128}
NUM_WARPS = 1


def run(a_offsets, a_chars, b_offsets, b_chars, match, n):
    if n == 0:
        return
    kernel[(n,)](
        a_offsets, a_chars, b_offsets, b_chars, match, n, **CONSTANTS, num_warps=NUM_WARPS
    )
