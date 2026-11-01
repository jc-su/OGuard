import triton
import triton.language as tl
import torch

@triton.jit
def kernel(offsets_ptr, chars_ptr, idx_ptr, out_offsets_ptr, out_chars_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < m

    # Load indices
    src_idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    # Load source offsets
    src_start = tl.load(offsets_ptr + src_idx, mask=mask, other=0)
    src_end = tl.load(offsets_ptr + src_idx + 1, mask=mask, other=0)

    # Load destination offset
    dst_start = tl.load(out_offsets_ptr + offsets, mask=mask, other=0)

    # Copy strings one at a time (simplified)
    length = src_end - src_start
    for j in tl.range(0, 1000):  # Max string length
        char_idx = src_start + j
        dst_char_idx = dst_start + j
        char_val = tl.load(chars_ptr + char_idx)
        tl.store(out_chars_ptr + dst_char_idx, char_val, mask=(j < length) & mask)

TYPES = {
    "offsets_ptr": "*i32",
    "chars_ptr": "*u8",
    "idx_ptr": "*i32",
    "out_offsets_ptr": "*i32",
    "out_chars_ptr": "*u8",
    "n": "i64",
    "m": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    grid = ((m + 255) // 256,)
    kernel[grid](offsets, chars, idx, out_offsets, out_chars, n, m, num_warps=NUM_WARPS)
