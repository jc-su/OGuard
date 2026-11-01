import triton
import triton.language as tl
import torch

@triton.jit
def kernel(group_ptr, value_ptr, best_ptr, n, g, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    groups = tl.load(group_ptr + offsets, mask=mask, other=-1)
    values = tl.load(value_ptr + offsets, mask=mask, other=0)
    
    old_best = tl.load(best_ptr + groups, mask=mask, other=-1)
    old_best_val = tl.where(old_best == -1, -9223372036854775808, tl.load(value_ptr + old_best, mask=mask, other=0))
    
    should_update = values > old_best_val
    tl.atomic_min(best_ptr + groups, offsets, mask=should_update & mask)

TYPES = {
    "group_ptr": "*i32",
    "value_ptr": "*i64",
    "best_ptr": "*i32",
    "n": "i64",
    "g": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(group, value, best, n, g):
    grid = ((n + 255) // 256,)
    kernel[grid](group, value, best, n, g, num_warps=NUM_WARPS)
