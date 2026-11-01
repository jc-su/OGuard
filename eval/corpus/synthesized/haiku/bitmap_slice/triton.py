import triton
import triton.language as tl
import torch

@triton.jit
def kernel(source_ptr, out_ptr, n, offset, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    src_byte_idx = (offset + offsets) // 8
    src_bit_idx = (offset + offsets) % 8
    src_bytes = tl.load(source_ptr + src_byte_idx, mask=mask, other=0)
    src_bits = (src_bytes >> src_bit_idx) & 1
    
    out_byte_idx = offsets // 8
    out_bit_idx = offsets % 8
    out_bytes = tl.load(out_ptr + out_byte_idx, mask=mask, other=0)
    out_bytes = tl.where(src_bits != 0, out_bytes | (1 << out_bit_idx), out_bytes & ~(1 << out_bit_idx))
    tl.store(out_ptr + out_byte_idx, out_bytes, mask=mask)

TYPES = {
    "source_ptr": "*u8",
    "out_ptr": "*u8",
    "n": "i64",
    "offset": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(source, out, n, offset):
    out.zero_()
    grid = ((n + 255) // 256,)
    kernel[grid](source, out, n, offset, num_warps=NUM_WARPS)
