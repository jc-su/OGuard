import triton
import triton.language as tl
import torch

@triton.jit
def kernel(offsets_ptr, chars_ptr, out_ptr, valid_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    start = tl.load(offsets_ptr + offsets, mask=mask, other=0)
    end = tl.load(offsets_ptr + offsets + 1, mask=mask, other=0)
    
    result = 0
    is_valid = True
    is_negative = False
    
    for j in range(1000):
        if j >= 1000:
            break
        c = tl.load(chars_ptr + start + j, mask=(start + j < end) & mask, other=0)
        
        if j == 0 and (c == ord('+') or c == ord('-')):
            is_negative = (c == ord('-'))
        else:
            if c >= ord('0') and c <= ord('9'):
                result = result * 10 + (c - ord('0'))
            else:
                is_valid = False
                break
    
    out = tl.where(is_negative, -result, result)
    valid_bit = tl.where(is_valid, 1, 0)
    
    byte_idx = offsets // 8
    bit_idx = offsets % 8
    valid_byte = tl.load(valid_ptr + byte_idx, mask=mask, other=0)
    valid_byte = tl.where(valid_bit, valid_byte | (1 << bit_idx), valid_byte)
    tl.store(valid_ptr + byte_idx, valid_byte, mask=mask)
    
    tl.store(out_ptr + offsets, out, mask=mask)

TYPES = {
    "offsets_ptr": "*i32",
    "chars_ptr": "*u8",
    "out_ptr": "*i64",
    "valid_ptr": "*u8",
    "n": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(offsets, chars, out, valid, n):
    valid.zero_()
    grid = ((n + 255) // 256,)
    kernel[grid](offsets, chars, out, valid, n, num_warps=NUM_WARPS)
