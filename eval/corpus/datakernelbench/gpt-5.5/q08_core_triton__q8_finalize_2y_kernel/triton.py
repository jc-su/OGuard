import triton
import triton.language as tl

TYPES = {
    "agg_ptr": "*fp64",
    "out_ptr": "*fp64",
}

CONSTANTS = {}

NUM_WARPS = 4


@triton.jit
def kernel(
    agg_ptr,
    out_ptr,
):
    brazil_1995 = tl.load(agg_ptr + 0)
    total_1995 = tl.load(agg_ptr + 1)
    brazil_1996 = tl.load(agg_ptr + 2)
    total_1996 = tl.load(agg_ptr + 3)

    tl.store(out_ptr + 0, brazil_1995 / total_1995)
    tl.store(out_ptr + 1, brazil_1996 / total_1996)
