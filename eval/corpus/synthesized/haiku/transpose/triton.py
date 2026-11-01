# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(source_ptr, out_ptr, rows, cols):
    # Process elements in a 2D grid
    pidx = tl.program_id(0)  # Row index
    pidy = tl.program_id(1)  # Column index

    if pidx < rows and pidy < cols:
        # Read from source[pidx * cols + pidy]
        source_pos = pidx * cols + pidy
        value = tl.load(source_ptr + source_pos)

        # Write to out[pidy * rows + pidx]
        out_pos = pidy * rows + pidx
        tl.store(out_ptr + out_pos, value)


TYPES = {
    "source_ptr": "*i32",
    "out_ptr": "*i32",
    "rows": "i64",
    "cols": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(source: torch.Tensor, out: torch.Tensor, rows: int, cols: int) -> None:
    """
    Launch the transpose kernel.

    Args:
        source: input matrix (int32, rows * cols), row-major
        out: output matrix (int32, rows * cols), row-major (transposed)
        rows: number of rows
        cols: number of columns
    """
    # Use a 2D grid: (rows, cols)
    grid = (rows, cols)
    kernel[grid](source, out, rows, cols, num_warps=NUM_WARPS)
