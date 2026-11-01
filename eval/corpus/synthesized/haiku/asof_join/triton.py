import triton
import triton.language as tl
import torch

@triton.jit
def kernel(left_ptr, right_ptr, out_ptr, n, r, tolerance, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    left_val = tl.load(left_ptr + offsets, mask=mask, other=0)

    best_idx = -1
    for j in range(10000):  # Max r, but will exit early
        if j >= r:
            break
        right_val = tl.load(right_ptr + j)
        matches = (right_val <= left_val) & (left_val - right_val <= tolerance)
        best_idx = tl.where(matches, j, best_idx)

    tl.store(out_ptr + offsets, best_idx, mask=mask)

TYPES = {
    "left_ptr": "*i64",
    "right_ptr": "*i64",
    "out_ptr": "*i32",
    "n": "i64",
    "r": "i64",
    "tolerance": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(left, right, out, n, r, tolerance):
    grid = ((n + 255) // 256,)
    kernel[grid](left, right, out, n, r, tolerance, num_warps=NUM_WARPS)
