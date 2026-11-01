import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, first_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    segment_start = tl.load(first_ptr + offsets, mask=mask, other=0)
    max_val = tl.load(values_ptr + segment_start, mask=mask, other=0)
    
    for j in tl.range(0, 10000):
        if j >= 10000:
            break
        val = tl.load(values_ptr + segment_start + j, mask=(segment_start + j <= offsets) & mask, other=0)
        max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)

TYPES = {
    "values_ptr": "*i64",
    "first_ptr": "*i32",
    "out_ptr": "*i64",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(values, first, out, n):
    grid = ((n + 255) // 256,)
    kernel[grid](values, first, out, n, num_warps=NUM_WARPS)
