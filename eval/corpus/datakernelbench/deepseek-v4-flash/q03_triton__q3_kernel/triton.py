import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "o_orderdate_days_ptr": "*i32",
    "o_shippriority_ptr": "*i32",
    "shipdate_lo_days": "i32",
    "orderdate_hi_days": "i32",
    "out_rev_ptr": "*fp64",
    "out_lk_ptr": "*i64",
    "out_od_ptr": "*i32",
    "out_sp_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 16384,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr,
    l_shipdate_days_ptr, o_orderdate_days_ptr, o_shippriority_ptr,
    shipdate_lo_days, orderdate_hi_days,
    out_rev_ptr, out_lk_ptr, out_od_ptr, out_sp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_ship = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    o_order = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=orderdate_hi_days)

    keep = mask & (l_ship > shipdate_lo_days) & (o_order < orderdate_hi_days)

    ep = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    rev = tl.where(keep, ep * (1.0 - disc), 0.0)
    lk = tl.load(l_orderkey_ptr + offs, mask=mask, other=0)
    od = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=0)
    
    tl.store(out_rev_ptr + offs, rev, mask=mask)
    tl.store(out_lk_ptr + offs, lk, mask=mask)
    tl.store(out_od_ptr + offs, od, mask=mask)
    tl.store(out_sp_ptr + offs, sp, mask=mask)
