import triton
import triton.language as tl
import torch

@triton.jit
def kernel(a_offsets_ptr, a_chars_ptr, b_offsets_ptr, b_chars_ptr, out_offsets_ptr, out_chars_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_start = tl.load(a_offsets_ptr + offsets, mask=mask, other=0)
    a_end = tl.load(a_offsets_ptr + offsets + 1, mask=mask, other=0)
    b_start = tl.load(b_offsets_ptr + offsets, mask=mask, other=0)
    b_end = tl.load(b_offsets_ptr + offsets + 1, mask=mask, other=0)
    out_start = tl.load(out_offsets_ptr + offsets, mask=mask, other=0)
    
    a_len = a_end - a_start
    b_len = b_end - b_start
    
    for j in tl.range(0, 1000):
        if j >= 1000:
            break
        a_char = tl.load(a_chars_ptr + a_start + j, mask=(j < a_len) & mask, other=0)
        tl.store(out_chars_ptr + out_start + j, a_char, mask=(j < a_len) & mask)
    
    for j in tl.range(0, 1000):
        if j >= 1000:
            break
        b_char = tl.load(b_chars_ptr + b_start + j, mask=(j < b_len) & mask, other=0)
        tl.store(out_chars_ptr + out_start + a_len + j, b_char, mask=(j < b_len) & mask)

TYPES = {
    "a_offsets_ptr": "*i32",
    "a_chars_ptr": "*u8",
    "b_offsets_ptr": "*i32",
    "b_chars_ptr": "*u8",
    "out_offsets_ptr": "*i32",
    "out_chars_ptr": "*u8",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    grid = ((n + 255) // 256,)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, num_warps=NUM_WARPS)
