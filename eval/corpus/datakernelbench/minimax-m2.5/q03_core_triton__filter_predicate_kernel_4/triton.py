import triton
import triton.language as tl

TYPES = {
    "l_shipdate_ptr": "*i32",
    "o_orderdate_ptr": "*i32",
    "shipdate_lo": "i32",
    "orderdate_hi": "i32",
    "mask_ptr": "*i32",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_shipdate_ptr, o_orderdate_ptr,
    shipdate_lo, orderdate_hi,
    mask_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    l_shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=0)
    o_orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=0)
    keep = mask & (l_shipdate > shipdate_lo) & (o_orderdate < orderdate_hi)
    tl.store(mask_ptr + offs, tl.cast(keep, tl.int32))
