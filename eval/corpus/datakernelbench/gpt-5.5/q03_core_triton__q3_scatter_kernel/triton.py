import triton
import triton.language as tl

TYPES = {
    "l_orderkey_ptr": "*i64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "o_orderdate_days_ptr": "*i32",
    "o_shippriority_ptr": "*i32",
    "keep_ptr": "*u8",
    "block_offsets_ptr": "*i64",
    "out_lk_ptr": "*i64",
    "out_rev_ptr": "*fp64",
    "out_od_ptr": "*i32",
    "out_sp_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_orderkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    o_orderdate_days_ptr,
    o_shippriority_ptr,
    keep_ptr,
    block_offsets_ptr,
    out_lk_ptr,
    out_rev_ptr,
    out_od_ptr,
    out_sp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    keep_u8 = tl.load(keep_ptr + offs, mask=in_bounds, other=0)
    keep = keep_u8 != 0

    local_rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    base = tl.load(block_offsets_ptr + pid)
    out_idx = base + local_rank

    lk = tl.load(l_orderkey_ptr + offs, mask=keep, other=0)
    od = tl.load(o_orderdate_days_ptr + offs, mask=keep, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=keep, other=0)
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    rev = price * (1.0 - disc)

    tl.store(out_lk_ptr + out_idx, lk, mask=keep)
    tl.store(out_rev_ptr + out_idx, rev, mask=keep)
    tl.store(out_od_ptr + out_idx, od, mask=keep)
    tl.store(out_sp_ptr + out_idx, sp, mask=keep)
