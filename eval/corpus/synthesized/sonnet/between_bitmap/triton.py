import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, valid, out, n, lo, hi, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = offs < n
    x = tl.load(values + offs, mask=inside, other=0)
    vbyte = tl.load(valid + (offs >> 3), mask=inside, other=0).to(tl.int32)
    bit = (vbyte >> (offs & 7).to(tl.int32)) & 1
    keep = inside & (bit != 0) & (x >= lo) & (x <= hi)
    # Eight rows make one output byte, least significant bit first.
    bits = tl.reshape(keep.to(tl.int32), (BLOCK // 8, 8))
    weight = 1 << tl.arange(0, 8)
    packed = tl.sum(bits * weight[None, :], axis=1)
    byte_offs = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8).to(tl.int64)
    tl.store(out + byte_offs, packed.to(tl.uint8), mask=byte_offs < (n + 7) // 8)


TYPES = {
    "values": "*i64",
    "valid": "*u8",
    "out": "*u8",
    "n": "i64",
    "lo": "i64",
    "hi": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, valid, out, n, lo, hi):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, valid, out, n, lo, hi, **CONSTANTS, num_warps=NUM_WARPS)
