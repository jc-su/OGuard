import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, out_ptr, overflow_ptr, n, shift, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    values = tl.load(values_ptr + offsets, mask=mask, other=0)
    
    if shift >= 0:
        power = 1
        for _ in range(int(shift)):
            power *= 10
        result = values * power
        out = result
        overflow = tl.zeros_like(overflow_ptr[offsets])
    else:
        power = 1
        for _ in range(int(-shift)):
            power *= 10
        abs_v = tl.abs(values)
        rounded = (abs_v + power // 2) // power
        out = tl.where(values < 0, -rounded, rounded)
        overflow = tl.zeros_like(overflow_ptr[offsets])
    
    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(overflow_ptr + offsets, overflow, mask=mask)

TYPES = {
    "values_ptr": "*i64",
    "out_ptr": "*i64",
    "overflow_ptr": "*i8",
    "n": "i64",
    "shift": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(values, out, overflow, n, shift):
    grid = ((n + 255) // 256,)
    kernel[grid](values, out, overflow, n, shift, num_warps=NUM_WARPS)
