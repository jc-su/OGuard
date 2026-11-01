import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, row_ptr, n):
    if tl.program_id(0) == 0 and n > 0:
        min_val = tl.load(values_ptr)
        min_row = 0
        
        for j in range(1, 10000000):
            if j >= n:
                break
            val = tl.load(values_ptr + j)
            if val < min_val:
                min_val = val
                min_row = j
        
        tl.store(row_ptr, min_row)

TYPES = {
    "values_ptr": "*i64",
    "row_ptr": "*i32",
    "n": "i64"
}

CONSTANTS = {}

NUM_WARPS = 1

def run(values, row, n):
    grid = (1,)
    kernel[grid](values, row, n, num_warps=NUM_WARPS)
