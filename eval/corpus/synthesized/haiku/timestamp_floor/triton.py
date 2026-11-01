import triton
import triton.language as tl
import torch

@triton.jit
def kernel(ts_ptr, out_ptr, n, unit, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    ts_val = tl.load(ts_ptr + offsets, mask=mask, other=0)
    
    result = tl.where(ts_val >= 0, (ts_val // unit) * unit, ((ts_val // unit) - 1) * unit)
    
    tl.store(out_ptr + offsets, result, mask=mask)

TYPES = {
    "ts_ptr": "*i64",
    "out_ptr": "*i64",
    "n": "i64",
    "unit": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(ts, out, n, unit):
    grid = ((n + 255) // 256,)
    kernel[grid](ts, out, n, unit, num_warps=NUM_WARPS)
