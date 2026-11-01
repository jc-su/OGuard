import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_ptr": "*i32",
    "date_lo": "i32",
    "date_hi": "i32",
    "revenue_ptr": "*fp64",
    "present_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_ptr,
    date_lo,
    date_hi,
    revenue_ptr,
    present_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate first to evaluate predicate
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Conditionally load expensive columns (selectivity is ~3.5%)
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    # Calculate revenue
    revenue = price * (1.0 - discount)
    
    # Scatter add to global accumulator array indexed directly by suppkey
    revenue_ptr_offset = revenue_ptr + suppkey
    present_ptr_offset = present_ptr + suppkey
    
    tl.atomic_add(revenue_ptr_offset, revenue, mask=keep)
    tl.atomic_max(present_ptr_offset, 1, mask=keep)
