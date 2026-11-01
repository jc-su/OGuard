"""run_end_encode: Arrow run-end encoding of values. count[0] = number of runs r; for
run j, run_ends[j] = row after its last row and run_values[j] = its value.

Row i ends a run when i == n - 1 or values[i] != values[i + 1]; run j is the run
ending at the (j + 1)-th such row. The output position of a run end depends on all
earlier rows, and Triton has no grid-wide barrier and the kernel has no scratch
memory to pass block totals between programs, so one program scans the column in
tiles of BLOCK rows, carrying the number of runs found so far. Within a tile the
positions come from a cumulative sum of the run-end flags.
"""

import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 16

TYPES = {
    "values": "*i64",
    "run_ends": "*i32",
    "run_values": "*i64",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(values, run_ends, run_values, count, n, BLOCK: tl.constexpr):
    n = tl.cast(n, tl.int64)
    lane = tl.arange(0, BLOCK)
    runs = tl.zeros((), tl.int64)
    for t0 in range(0, n, BLOCK):
        rows = t0 + lane
        in_range = rows < n
        v = tl.load(values + rows, mask=in_range, other=0)
        following = tl.load(values + rows + 1, mask=rows + 1 < n, other=0)
        flag = in_range & ((rows == n - 1) | (v != following))
        f = flag.to(tl.int64)
        pos = runs + tl.cumsum(f, 0) - 1
        tl.store(run_ends + pos, (rows + 1).to(tl.int32), mask=flag)
        tl.store(run_values + pos, v, mask=flag)
        runs += tl.sum(f, 0)
    tl.store(count, runs.to(tl.int32))


def run(values, run_ends, run_values, count, n):
    kernel[(1,)](values, run_ends, run_values, count, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
