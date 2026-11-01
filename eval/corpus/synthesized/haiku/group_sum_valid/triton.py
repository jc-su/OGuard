import triton
import triton.language as tl
import torch

@triton.jit
def kernel(group_ptr, value_ptr, valid_ptr, counts_ptr, sums_ptr, n, g, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load values and groups
    values = tl.load(value_ptr + offsets, mask=mask, other=0)
    groups = tl.load(group_ptr + offsets, mask=mask, other=0)

    # Load validity bits
    byte_offsets = offsets // 8
    bit_offsets = offsets % 8
    valid_bytes = tl.load(valid_ptr + byte_offsets, mask=mask, other=0)
    valid_bits = (valid_bytes >> bit_offsets) & 1

    # Atomic operations
    valid_mask = (valid_bits != 0) & mask
    tl.atomic_add(counts_ptr + groups, 1, mask=valid_mask)
    tl.atomic_add(sums_ptr + groups, values, mask=valid_mask)

TYPES = {
    "group_ptr": "*i32",
    "value_ptr": "*i64",
    "valid_ptr": "*u8",
    "counts_ptr": "*i64",
    "sums_ptr": "*i64",
    "n": "i64",
    "g": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(group, value, valid, counts, sums, n, g):
    counts.zero_()
    sums.zero_()
    grid = ((n + 255) // 256,)
    kernel[grid](group, value, valid, counts, sums, n, g, num_warps=NUM_WARPS)
