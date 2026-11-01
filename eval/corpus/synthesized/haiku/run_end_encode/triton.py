import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, run_ends_ptr, run_values_ptr, count_ptr, n):
    if tl.program_id(0) == 0 and n > 0:
        run_count = 0
        j = 0
        
        while j < n:
            current_val = tl.load(values_ptr + j)
            run_start = j
            
            while j < n and tl.load(values_ptr + j) == current_val:
                j += 1
            
            tl.store(run_ends_ptr + run_count, j)
            tl.store(run_values_ptr + run_count, current_val)
            run_count += 1
        
        tl.store(count_ptr, run_count)

TYPES = {
    "values_ptr": "*i64",
    "run_ends_ptr": "*i32",
    "run_values_ptr": "*i64",
    "count_ptr": "*i32",
    "n": "i64"
}

CONSTANTS = {}

NUM_WARPS = 1

def run(values, run_ends, run_values, count, n):
    grid = (1,)
    kernel[grid](values, run_ends, run_values, count, n, num_warps=NUM_WARPS)
