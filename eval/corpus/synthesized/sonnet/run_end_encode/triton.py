import torch
import triton
import triton.language as tl


# One program walks the column tile by tile. The inclusive scan of the run-start
# flags, plus a carry over the tiles, gives every run its output position.
@triton.jit
def kernel(values, run_ends, run_values, count, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK).to(tl.int64)
    carry = tl.cast(0, tl.int64)
    for start in range(0, n, BLOCK):
        offs = start + lane
        inside = offs < n
        x = tl.load(values + offs, mask=inside, other=0)
        prev = tl.load(values + offs - 1, mask=inside & (offs > 0), other=0)
        is_start = inside & ((offs == 0) | (x != prev))
        flag = is_start.to(tl.int32)
        pos = carry + tl.cumsum(flag, axis=0).to(tl.int64) - flag.to(tl.int64)
        # The run that starts at row i has index pos and closes the run pos - 1.
        tl.store(run_values + pos, x, mask=is_start)
        tl.store(run_ends + (pos - 1), offs.to(tl.int32), mask=is_start & (pos > 0))
        carry += tl.sum(flag, axis=0).to(tl.int64)
    # The last run ends at row n.
    tl.store(run_ends + (carry - 1), tl.cast(n, tl.int32), mask=n > 0)
    tl.store(count, carry.to(tl.int32), mask=n > 0)


TYPES = {
    "values": "*i64",
    "run_ends": "*i32",
    "run_values": "*i64",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, run_ends, run_values, count, n):
    kernel[(1,)](values, run_ends, run_values, count, n, **CONSTANTS, num_warps=NUM_WARPS)
