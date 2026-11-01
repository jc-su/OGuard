import triton
import triton.language as tl
import torch

@triton.jit
def kernel(ts_ptr, values_ptr, out_ptr, n, range_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    ts_i = tl.load(ts_ptr + offsets, mask=mask, other=0)
    lower_bound = ts_i - range_val
    
    total = 0
    for j in range(10000):
        if j >= n:
            break
        ts_j = tl.load(ts_ptr + j)
        values_j = tl.load(values_ptr + j)
        in_range = (ts_j >= lower_bound) & (ts_j <= ts_i)
        total = tl.where(in_range, total + values_j, total)
    
    tl.store(out_ptr + offsets, total, mask=mask)

TYPES = {
    "ts_ptr": "*i64",
    "values_ptr": "*i64",
    "out_ptr": "*i64",
    "n": "i64",
    "range_val": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(ts, values, out, n, range_val):
    grid = ((n + 255) // 256,)
    kernel[grid](ts, values, out, n, range_val, num_warps=NUM_WARPS)
