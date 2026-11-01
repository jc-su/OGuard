import torch
import triton
import triton.language as tl


@triton.jit
def kernel(valid, count, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nbytes = (n + 7) // 8
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nbytes
    byte = tl.load(valid + offs, mask=mask, other=0).to(tl.int32)
    bits = tl.minimum(n - offs * 8, 8).to(tl.int32)
    bits = tl.where(mask, bits, 0)
    live = byte & ((1 << bits) - 1)
    live = live - ((live >> 1) & 0x55)
    live = (live & 0x33) + ((live >> 2) & 0x33)
    live = (live + (live >> 4)) & 0x0F
    zeros = bits - live
    tl.atomic_add(count, tl.sum(zeros, 0).to(tl.int64))


TYPES = {"valid": "*u8", "count": "*i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 4


def run(valid, count, n):
    grid = (max(1, triton.cdiv((n + 7) // 8, CONSTANTS["BLOCK"])),)
    kernel[grid](valid, count, n, **CONSTANTS, num_warps=NUM_WARPS)
