import triton
import triton.language as tl

TYPES = {
    "sd_ptr": "*i32",
    "od_ptr": "*i32",
    "block_counts_ptr": "*i32",
    "sd_lo": "i32",
    "od_hi": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    sd_ptr, od_ptr,
    block_counts_ptr,
    sd_lo, od_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    sd = tl.load(sd_ptr + offs, mask=mask, other=sd_lo - 1)
    od = tl.load(od_ptr + offs, mask=mask, other=od_hi + 1)
    keep = (sd > sd_lo) & (od < od_hi) & mask
    count = tl.sum(keep.to(tl.int32))
    tl.store(block_counts_ptr + pid, count)
