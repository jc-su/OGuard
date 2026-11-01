import triton
import triton.language as tl
import torch

@triton.jit
def kernel(values_ptr, valid_ptr, out_ptr, n, lo, hi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load values and validity bits
    values = tl.load(values_ptr + offsets, mask=mask, other=0)

    # Load validity bits - need to handle bitmap
    byte_offsets = offsets // 8
    bit_offsets = offsets % 8
    valid_bytes = tl.load(valid_ptr + byte_offsets, mask=mask, other=0)
    valid_bits = (valid_bytes >> bit_offsets) & 1

    # Check range condition
    in_range = (values >= lo) & (values <= hi)
    result_bits = valid_bits & in_range

    # Write back to output bitmap
    # Since we're doing element-wise updates to a bitmap, we need to be careful
    # For now, assume out is pre-zeroed and we just need to set bits
    out_bytes = tl.load(out_ptr + byte_offsets, mask=mask, other=0)
    out_bytes = tl.where(result_bits != 0, out_bytes | (1 << bit_offsets), out_bytes & ~(1 << bit_offsets))
    tl.store(out_ptr + byte_offsets, out_bytes, mask=mask)

TYPES = {
    "values_ptr": "*i64",
    "valid_ptr": "*u8",
    "out_ptr": "*u8",
    "n": "i64",
    "lo": "i64",
    "hi": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(values, valid, out, n, lo, hi):
    out.zero_()
    grid = ((n + 255) // 256,)
    kernel[grid](values, valid, out, n, lo, hi, num_warps=NUM_WARPS)
