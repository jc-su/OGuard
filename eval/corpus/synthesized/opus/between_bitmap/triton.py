"""between_bitmap: out bit i = valid(i) and lo <= values[i] <= hi, as an Arrow bitmap.

Each program evaluates BLOCK consecutive rows starting at a multiple of BLOCK (a
multiple of 8) and writes their BLOCK // 8 output bytes, so each byte has one
writer and the bits at positions n and above are 0.
"""

import triton
import triton.language as tl

BLOCK = 2048
NUM_WARPS = 4

TYPES = {
    "values": "*i64",
    "valid": "*u8",
    "out": "*u8",
    "n": "i64",
    "lo": "i64",
    "hi": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(values, valid, out, n, lo, hi, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    v = tl.load(values + rows, mask=in_range, other=0)
    vbyte = tl.load(valid + (rows >> 3), mask=in_range, other=0).to(tl.int32)
    is_valid = ((vbyte >> (rows & 7).to(tl.int32)) & 1) != 0
    bit = (in_range & is_valid & (v >= lo) & (v <= hi)).to(tl.int32)
    bits = tl.reshape(bit, (BLOCK // 8, 8)) << tl.arange(0, 8)[None, :]
    byte_val = tl.sum(bits, axis=1)
    out_idx = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    tl.store(out + out_idx, byte_val.to(tl.uint8), mask=out_idx < ((n + 7) >> 3))


def run(values, valid, out, n, lo, hi):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](values, valid, out, n, lo, hi, BLOCK=BLOCK, num_warps=NUM_WARPS)
