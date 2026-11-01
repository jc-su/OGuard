import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, part, counts, n, p, BLOCK: tl.constexpr, BINS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(keys + offs, mask=mask, other=0).to(tl.uint64, bitcast=True)
    h = (x * 0x9E3779B97F4A7C15) >> 32
    pt = (h.to(tl.int64) % p).to(tl.int32)
    tl.store(part + offs, pt, mask=mask)
    hist = tl.histogram(tl.where(mask, pt, 0), BINS)
    bins = tl.arange(0, BINS)
    padding = BLOCK - tl.sum(mask.to(tl.int32), 0)
    hist = hist - tl.where(bins == 0, padding, 0)
    tl.atomic_add(counts + bins, hist, mask=(bins < p) & (hist != 0))


TYPES = {"keys": "*i64", "part": "*i32", "counts": "*i32", "n": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": 4096, "BINS": 1024}
NUM_WARPS = 8


def run(keys, part, counts, n, p):
    assert 1 <= p <= CONSTANTS["BINS"]
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](keys, part, counts, n, p, **CONSTANTS, num_warps=NUM_WARPS)
