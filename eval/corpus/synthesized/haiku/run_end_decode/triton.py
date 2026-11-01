import triton
import triton.language as tl
import torch


@triton.jit
def kernel(run_values, run_ends, out, r, n, BLOCK_SIZE: tl.constexpr):
    """
    Decode run-end encoded column.
    run_values: float64 array of r values
    run_ends: int32 array of r end positions (strictly increasing, run_ends[r-1] = n)
    out: float64 output array of n values
    r: number of runs
    n: total number of rows
    """
    # Grid indexed by output position
    idx = tl.program_id(0) * BLOCK_SIZE

    # For each element in this block
    for i in range(BLOCK_SIZE):
        pos = idx + i
        if pos < n:
            # Find which run this position belongs to
            # We want: the unique run_id where start <= pos < end
            # where start = run_ends[run_id-1] (or 0 if run_id == 0)
            run_id = 0
            for r_idx in range(r):
                curr_end = tl.load(run_ends + r_idx)
                if r_idx == 0:
                    prev_end = 0
                else:
                    prev_end = tl.load(run_ends + r_idx - 1)

                # Check if pos is in range [prev_end, curr_end)
                in_range = (pos >= prev_end) & (pos < curr_end)
                if in_range:
                    run_id = r_idx

            # Get the value for this run
            value = tl.load(run_values + run_id)
            # Store the value
            tl.store(out + pos, value, eviction_policy='evict_last')


TYPES = {
    "run_values": "*fp64",
    "run_ends": "*i32",
    "out": "*fp64",
    "r": "i64",
    "n": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(run_values, run_ends, out, r, n):
    """
    run_values: torch.tensor(shape=(r,), dtype=torch.float64)
    run_ends: torch.tensor(shape=(r,), dtype=torch.int32)
    out: torch.tensor(shape=(n,), dtype=torch.float64)
    r: int (number of runs)
    n: int (total number of rows)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](run_values, run_ends, out, r, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
