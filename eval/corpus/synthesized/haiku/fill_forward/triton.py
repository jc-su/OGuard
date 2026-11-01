import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, valid_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    last_valid = 0
    found = False
    
    for j in tl.range(0, 10000000):
        if j >= n:
            break
        byte_idx = j // 8
        bit_idx = j % 8
        valid_byte = tl.load(valid_ptr + byte_idx)
        valid_bit = (valid_byte >> bit_idx) & 1
        
        if valid_bit:
            last_valid = tl.load(values_ptr + j)
            found = True
        
        if j >= offsets:
            break
    
    result = tl.where(found, last_valid, 0)
    tl.store(out_ptr + offsets, result, mask=mask)

TYPES = {
    "values_ptr": "*i64",
    "valid_ptr": "*u8",
    "out_ptr": "*i64",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(values, valid, out, n):
    grid = ((n + 255) // 256,)
    kernel[grid](values, valid, out, n, num_warps=NUM_WARPS)
