# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace
# by the test/loader, to avoid shadowing conflicts

@triton.jit
def kernel(values_ptr, keep_ptr, out_ptr, count_ptr, n):
    # Each program instance handles one element
    idx = tl.program_id(0)

    if idx < n:
        # Load the keep flag for this element
        keep = tl.load(keep_ptr + idx)

        if keep != 0:
            # Atomically increment count to get position
            pos = tl.atomic_add(count_ptr, 1)

            # Load and write the value
            value = tl.load(values_ptr + idx)
            tl.store(out_ptr + pos, value)


TYPES = {
    "values_ptr": "*i64",
    "keep_ptr": "*i8",
    "out_ptr": "*i64",
    "count_ptr": "*i32",
    "n": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(values: torch.Tensor, keep: torch.Tensor, out: torch.Tensor, count: torch.Tensor, n: int) -> None:
    """
    Launch the stream_compact kernel.

    Args:
        values: input values (int64, n)
        keep: keep mask (int8, n)
        out: output array (int64, n)
        count: counter (int32, 1)
        n: number of elements
    """
    if n == 0:
        return

    grid = (n,)
    kernel[grid](values, keep, out, count, n, num_warps=NUM_WARPS)
