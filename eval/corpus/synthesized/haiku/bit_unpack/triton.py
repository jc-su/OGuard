import triton
import triton.language as tl
import torch

@triton.jit
def kernel(packed_ptr, out_ptr, n, bits, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    bit_pos = offsets * bits
    
    code = 0
    for b in tl.range(0, 32):
        if b >= bits:
            break
        pos = bit_pos + b
        w_idx = pos // 32
        b_idx = pos % 32
        word = tl.load(packed_ptr + w_idx, mask=mask, other=0)
        bit = (word >> b_idx) & 1
        code = tl.where(mask, code | (bit << b), 0)
    
    tl.store(out_ptr + offsets, code, mask=mask)

TYPES = {
    "packed_ptr": "*u32",
    "out_ptr": "*u32",
    "n": "i64",
    "bits": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(packed, out, n, bits):
    grid = ((n + 255) // 256,)
    kernel[grid](packed, out, n, bits, num_warps=NUM_WARPS)
