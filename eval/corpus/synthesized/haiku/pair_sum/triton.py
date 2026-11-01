# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(left_idx_ptr, right_idx_ptr, left_ptr, right_ptr, out_ptr, m, l, r):
    # Each program processes one pair
    idx = tl.program_id(0)

    if idx < m:
        # Load indices
        left_i = tl.load(left_idx_ptr + idx)
        right_i = tl.load(right_idx_ptr + idx)

        # Load values
        left_val = tl.load(left_ptr + left_i)
        right_val = tl.load(right_ptr + right_i)

        # Compute sum
        result = left_val + right_val

        # Store result
        tl.store(out_ptr + idx, result)


TYPES = {
    "left_idx_ptr": "*i32",
    "right_idx_ptr": "*i32",
    "left_ptr": "*f64",
    "right_ptr": "*f64",
    "out_ptr": "*f64",
    "m": "i64",
    "l": "i64",
    "r": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(left_idx: torch.Tensor, right_idx: torch.Tensor, left: torch.Tensor, right: torch.Tensor, out: torch.Tensor, m: int, l: int, r: int) -> None:
    """
    Launch the pair_sum kernel.

    Args:
        left_idx: indices into left array (int32, m)
        right_idx: indices into right array (int32, m)
        left: left array (float64, l)
        right: right array (float64, r)
        out: output array (float64, m)
        m: number of pairs
        l: length of left array
        r: length of right array
    """
    if m == 0:
        return

    grid = (m,)
    kernel[grid](left_idx, right_idx, left, right, out, m, l, r, num_warps=NUM_WARPS)
