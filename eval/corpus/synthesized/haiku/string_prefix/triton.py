# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(offsets_ptr, chars_ptr, out_ptr, n, k):
    # Get the string index and byte index
    pidx = tl.program_id(0)  # String index
    pidy = tl.program_id(1)  # Byte index

    if pidx < n and pidy < k:
        # Get the start and end offsets for this string
        start = tl.load(offsets_ptr + pidx)
        end = tl.load(offsets_ptr + pidx + 1)

        # Length of this string
        length = end - start

        # Output position
        out_pos = pidx * k + pidy

        # Check if this byte exists in the string
        if pidy < length:
            # Read from chars
            char_byte = tl.load(chars_ptr + start + pidy)
            tl.store(out_ptr + out_pos, char_byte)
        else:
            # Pad with zero
            tl.store(out_ptr + out_pos, 0)


TYPES = {
    "offsets_ptr": "*i32",
    "chars_ptr": "*u8",
    "out_ptr": "*u8",
    "n": "i64",
    "k": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(offsets: torch.Tensor, chars: torch.Tensor, out: torch.Tensor, n: int, k: int) -> None:
    """
    Launch the string_prefix kernel.

    Args:
        offsets: string offsets (int32, n + 1)
        chars: character data (uint8, offsets[n])
        out: output array (uint8, n * k)
        n: number of strings
        k: prefix length
    """
    if n == 0 or k == 0:
        return

    # Use a 2D grid: (n, k)
    # But Triton expects grid to be a tuple of integers, not a 2D grid directly
    # For 2D grids, we use program_id(0) and program_id(1)
    # The grid is specified as (grid_x, grid_y)
    grid = (n, k)
    kernel[grid](offsets, chars, out, n, k, num_warps=NUM_WARPS)
