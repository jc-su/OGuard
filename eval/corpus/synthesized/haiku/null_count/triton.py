# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(valid_ptr, count_ptr, n):
    # Each program processes one byte of the bitmap
    idx = tl.program_id(0)

    # Number of bytes in the bitmap
    num_bytes = (n + 7) >> 3  # (n + 7) / 8

    if idx < num_bytes:
        # Load the byte
        byte_val = tl.load(valid_ptr + idx)

        # Count how many bits are 0 (null)
        # For this byte, we need to check which of its 8 bits are 0
        # and count them, being careful about the last byte

        # Determine which bits in this byte are valid rows
        start_row = idx * 8
        end_row = tl.minimum(start_row + 8, n)
        rows_in_byte = end_row - start_row

        # Count null bits (0s) in the valid rows
        null_count = 0
        for bit_idx in range(8):
            if bit_idx < rows_in_byte:
                # Check if bit is set (1 = valid, 0 = null)
                bit_val = (byte_val >> bit_idx) & 1
                if bit_val == 0:
                    null_count += 1

        # Atomically add to the count
        if null_count > 0:
            tl.atomic_add(count_ptr, null_count)


TYPES = {
    "valid_ptr": "*u8",
    "count_ptr": "*i64",
    "n": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(valid: torch.Tensor, count: torch.Tensor, n: int) -> None:
    """
    Launch the null_count kernel.

    Args:
        valid: Arrow validity bitmap (uint8, (n + 7) / 8)
        count: output counter (int64, 1), zero on entry
        n: number of rows
    """
    if n == 0:
        return

    # Number of bytes in the bitmap
    num_bytes = (n + 7) >> 3
    if num_bytes == 0:
        return

    grid = (num_bytes,)
    kernel[grid](valid, count, n, num_warps=NUM_WARPS)
