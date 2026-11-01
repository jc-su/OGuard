import triton
import triton.language as tl
import torch

@triton.jit
def kernel(offsets_ptr, chars_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    start = tl.load(offsets_ptr + offsets, mask=mask, other=0)
    end = tl.load(offsets_ptr + offsets + 1, mask=mask, other=0)
    
    h = 2166136261
    for j in range(1000):
        if j >= 1000:
            break
        char_idx = start + j
        char_val = tl.load(chars_ptr + char_idx)
        h = ((h ^ char_val) * 16777619) & 0xFFFFFFFF
        if j >= (end - start):
            break
    
    tl.store(out_ptr + offsets, h, mask=mask)

TYPES = {
    "offsets_ptr": "*i32",
    "chars_ptr": "*u8",
    "out_ptr": "*u32",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(offsets, chars, out, n):
    grid = ((n + 255) // 256,)
    kernel[grid](offsets, chars, out, n, num_warps=NUM_WARPS)
