import triton
import triton.language as tl
import torch

@triton.jit
def kernel(days_ptr, months_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    days = tl.load(days_ptr + offsets, mask=mask, other=0)
    months = tl.load(months_ptr + offsets, mask=mask, other=0)
    
    out = days + months * 30
    
    tl.store(out_ptr + offsets, out, mask=mask)

TYPES = {
    "days_ptr": "*i32",
    "months_ptr": "*i32",
    "out_ptr": "*i32",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(days, months, out, n):
    grid = ((n + 255) // 256,)
    kernel[grid](days, months, out, n, num_warps=NUM_WARPS)
