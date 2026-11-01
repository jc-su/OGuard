import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_ptr": "*i32",
    "o_orderdate_ptr": "*i32",
    "o_shippriority_ptr": "*i32",
    "shipdate_lo": "i32",
    "orderdate_hi": "i32",
    "n_elements": "i32",
    "out_lk_ptr": "*i64",
    "out_od_ptr": "*i64",
    "out_sp_ptr": "*i64",
    "out_rev_ptr": "*fp64",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr, l_shipdate_ptr,
    o_orderdate_ptr, o_shippriority_ptr,
    shipdate_lo, orderdate_hi,
    n_elements,
    out_lk_ptr, out_od_ptr, out_sp_ptr, out_rev_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all columns
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=shipdate_lo)
    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=orderdate_hi)
    lk = tl.load(l_orderkey_ptr + offs, mask=mask, other=-1)
    price = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=-1)

    # Apply predicates
    keep = mask & (shipdate > shipdate_lo) & (orderdate < orderdate_hi)

    # Store kept values for later aggregation
    base = tl.program_id(0) * BLOCK_SIZE
    tl.store(out_lk_ptr + base + tl.arange(0, BLOCK_SIZE), lk, mask=keep)
    tl.store(out_od_ptr + base + tl.arange(0, BLOCK_SIZE), orderdate, mask=keep)
    tl.store(out_sp_ptr + base + tl.arange(0, BLOCK_SIZE), sp, mask=keep)
    tl.store(out_rev_ptr + base + tl.arange(0, BLOCK_SIZE),
             tl.where(keep, price * (1.0 - discount), 0.0), mask=keep)
