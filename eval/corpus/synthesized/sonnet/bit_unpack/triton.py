import torch
import triton
import triton.language as tl


@triton.jit
def kernel(packed, out, n, w, bits, BLOCK: tl.constexpr):
    # One lane per code. A code spans two words when its bits cross a word boundary.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = i < n
    p = i * bits
    word = p >> 5
    shift = p & 31
    spans = inside & (shift + bits > 32)
    lo = tl.load(packed + word, mask=inside, other=0).to(tl.uint64)
    hi = tl.load(packed + word + 1, mask=spans, other=0).to(tl.uint64)
    one = tl.full([BLOCK], 1, tl.uint64)
    v = (lo | (hi << tl.full([BLOCK], 32, tl.uint64))) >> shift.to(tl.uint64)
    mask = (one << tl.full([BLOCK], bits, tl.uint64)) - one
    tl.store(out + i, (v & mask).to(tl.uint32), mask=inside)


TYPES = {"packed": "*u32", "out": "*u32", "n": "i64", "w": "i64", "bits": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(packed, out, n, w, bits):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](packed, out, n, w, bits, **CONSTANTS, num_warps=NUM_WARPS)
