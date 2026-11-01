import triton
import triton.language as tl

TYPES = {
    "lk_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "l_ship_ptr": "*i32",
    "o_date_ptr": "*i32",
    "sp_ptr": "*i32",
    "shipdate_lo": "i32",
    "orderdate_hi": "i32",
    "out_lk_ptr": "*i64",
    "out_rev_ptr": "*fp64",
    "out_od_ptr": "*i32",
    "out_sp_ptr": "*i32",
    "cnt_ptr": "*i64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 2048}
NUM_WARPS = 8


@triton.jit
def kernel(
    # input columns
    lk_ptr,                 # int64 *
    price_ptr,              # float64 *
    discount_ptr,           # float64 *
    l_ship_ptr,             # int32 *
    o_date_ptr,             # int32 *
    sp_ptr,                 # int32 *
    # scalar predicates
    shipdate_lo,            # int32
    orderdate_hi,           # int32
    # output buffers (pre‑allocated, size >= input rows)
    out_lk_ptr,             # int64 *
    out_rev_ptr,            # float64 *
    out_od_ptr,             # int32 *
    out_sp_ptr,             # int32 *
    # counter for total kept rows
    cnt_ptr,                # int64 *
    # runtime
    n_elements,             # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns needed for predicate
    l_ship = tl.load(l_ship_ptr + offs, mask=mask, other=0)
    o_date = tl.load(o_date_ptr + offs, mask=mask, other=0)

    # Predicate: l_shipdate > shipdate_lo AND o_orderdate < orderdate_hi
    keep = mask & (l_ship > shipdate_lo) & (o_date < orderdate_hi)

    # Compute revenue for kept rows
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    rev = price * (1.0 - discount)                     # float64

    # ---------- compact write index ----------
    keep_i = tl.where(keep, 1, 0).to(tl.int64)        # int64 flags per lane
    pref = tl.cumsum(keep_i, axis=0) - keep_i         # exclusive prefix sum
    block_sum = tl.sum(keep_i, axis=0)                # total kept in this block

    # Lane 0 atomically reserves space and receives previous total
    prev_total = tl.atomic_add(cnt_ptr, block_sum)    # scalar int64
    # Broadcast by adding scalar to vector prefix sum
    write_idx = prev_total + pref                     # global write positions

    # Load remaining columns for kept rows
    lk = tl.load(lk_ptr + offs, mask=keep, other=0)
    od = tl.load(o_date_ptr + offs, mask=keep, other=0)
    sp = tl.load(sp_ptr + offs, mask=keep, other=0)

    # Store compacted results
    tl.store(out_lk_ptr + write_idx, lk, mask=keep)
    tl.store(out_rev_ptr + write_idx, rev, mask=keep)
    tl.store(out_od_ptr + write_idx, od, mask=keep)
    tl.store(out_sp_ptr + write_idx, sp, mask=keep)
