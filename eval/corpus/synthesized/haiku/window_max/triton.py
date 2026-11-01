# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(values_ptr, out_ptr, n, w):
    # Each program processes one output element
    idx = tl.program_id(0)

    if idx < n:
        # Compute max of values[max(0, idx - w + 1) : idx + 1]
        start = tl.maximum(0, idx - w + 1)

        # Find the maximum value in the window
        max_val = tl.load(values_ptr + start)

        for i in range(start, idx + 1):
            val = tl.load(values_ptr + i)
            max_val = tl.maximum(max_val, val)

        tl.store(out_ptr + idx, max_val)


TYPES = {
    "values_ptr": "*i64",
    "out_ptr": "*i64",
    "n": "i64",
    "w": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(values: torch.Tensor, out: torch.Tensor, n: int, w: int) -> None:
    """
    Launch the window_max kernel.

    Args:
        values: input values (int64, n)
        out: output values (int64, n)
        n: number of elements
        w: window size
    """
    if n == 0:
        return

    grid = (n,)
    kernel[grid](values, out, n, w, num_warps=NUM_WARPS)
