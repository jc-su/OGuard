import triton
import triton.language as tl

TYPES = {
    "partial_hist_ptr": "*i64",
    "hist_ptr": "*i64",
}

CONSTANTS = {"num_blocks": 1465, "SMALL_BINS": 256, "REDUCE_BLOCK": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    partial_hist_ptr,
    hist_ptr,
    num_blocks: tl.constexpr,
    SMALL_BINS: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    bin_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    offs = chunk_id * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    mask = offs < num_blocks

    vals = tl.load(partial_hist_ptr + offs * SMALL_BINS + bin_id, mask=mask, other=0)
    total = tl.sum(vals)
    tl.atomic_add(hist_ptr + bin_id, total, sem="relaxed")
