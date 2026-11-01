import triton
import triton.language as tl

TYPES = {
    "l_shipdate_ptr": "*i32",
    "o_orderdate_ptr": "*i32",
    "shipdate_lo": "i32",
    "orderdate_hi": "i32",
    "out_valid_ptr": "*i8",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_shipdate_ptr, o_orderdate_ptr,
    shipdate_lo, orderdate_hi,
    out_valid_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=0)
    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=0)
    
    valid = (shipdate > shipdate_lo) & (orderdate < orderdate_hi)
    tl.store(out_valid_ptr + offs, valid.to(tl.int8), mask=mask)
