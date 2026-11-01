import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_ptr": "*i32",
    "o_orderdate_ptr": "*i32",
    "o_shippriority_ptr": "*i32",
    "rev_out_ptr": "*fp64",
    "shipdate_lo": "i32",
    "orderdate_hi": "i32",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr,
    l_shipdate_ptr, o_orderdate_ptr, o_shippriority_ptr,
    rev_out_ptr,
    shipdate_lo, orderdate_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_ship = tl.load(l_shipdate_ptr + offs, mask=mask, other=orderdate_hi)
    o_date = tl.load(o_orderdate_ptr + offs, mask=mask, other=shipdate_lo)

    keep = mask & (l_ship > shipdate_lo) & (o_date < orderdate_hi)

    lk = tl.load(l_orderkey_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int64))
    od = tl.load(o_orderdate_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int32))
    sp = tl.load(o_shippriority_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int32))
    ep = tl.load(l_extendedprice_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.float64))
    disc = tl.load(l_discount_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.float64))

    rev = tl.where(keep, ep * (1.0 - disc), tl.zeros([BLOCK_SIZE], dtype=tl.float64))

    # Store filtered outputs back (masked rows become 0)
    tl.store(l_orderkey_ptr + offs, tl.where(keep, lk, tl.zeros([BLOCK_SIZE], dtype=tl.int64)), mask=mask)
    tl.store(o_orderdate_ptr + offs, tl.where(keep, od, tl.zeros([BLOCK_SIZE], dtype=tl.int32)), mask=mask)
    tl.store(o_shippriority_ptr + offs, tl.where(keep, sp, tl.zeros([BLOCK_SIZE], dtype=tl.int32)), mask=mask)
    tl.store(rev_out_ptr + offs, rev, mask=mask)
