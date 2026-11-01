import triton
import triton.language as tl
import torch


@triton.jit
def kernel(valid, rows, count, n, BLOCK_SIZE: tl.constexpr):
    """
    Extract valid row indices from Arrow validity bitmap.
    valid: uint8 array of (n+7)/8 bytes
    rows: int32 output array of capacity n
    count: int32 scalar, counts valid rows
    """
    pid = tl.program_id(0)

    # Each thread processes one row
    for i in range(BLOCK_SIZE):
        row_id = pid * BLOCK_SIZE + i
        if row_id < n:
            byte_idx = row_id // 8
            bit_idx = row_id % 8
            
            # Calculate the size of the validity bitmap
            bitmap_size = (n + 7) // 8

            # Load the byte with bounds checking
            is_byte_in_bounds = byte_idx >= 0 and byte_idx < bitmap_size
            # If byte_idx is out of bounds, use 0 as the default value
            valid_byte = tl.where(is_byte_in_bounds, tl.load(valid + byte_idx, eviction_policy='evict_last'), 0)

            # Check if bit is set (LSB first)
            is_valid = (valid_byte >> bit_idx) & 1

            if is_valid:
                # Atomic increment count and get position
                pos = tl.atomic_add(count, 1)
                # Explicit bounds check on position before storing
                is_pos_valid = pos >= 0 and pos < n
                # Only store if position is valid
                tl.store(rows + tl.where(is_pos_valid, pos, 0), row_id, eviction_policy='evict_last', mask=is_pos_valid)


TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(valid, rows, count, n):
    """
    valid: torch.tensor(shape=(ceil(n/8),), dtype=torch.uint8)
    rows: torch.tensor(shape=(n,), dtype=torch.int32)
    count: torch.tensor(shape=(1,), dtype=torch.int32)
    n: int
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](valid, rows, count, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
