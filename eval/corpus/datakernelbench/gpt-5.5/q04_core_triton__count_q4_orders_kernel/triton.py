import triton
import triton.language as tl

TYPES = {
    "o_orderkey_ptr": "*i64",
    "o_orderdate_days_ptr": "*i32",
    "o_orderpriority_codes_ptr": "*i32",
    "flags_ptr": "*u8",
    "counts_ptr": "*i64",
    "n_elements": "i32",
    "date_lo_days": "i32",
    "date_hi_days": "i32",
    "flag_n": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024, "NUM_PRIORITIES": 5}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_orderpriority_codes_ptr,
    flags_ptr,
    counts_ptr,
    n_elements,
    date_lo_days,
    date_hi_days,
    flag_n,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRIORITIES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    order_days = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=date_hi_days)
    date_keep = mask & (order_days >= date_lo_days) & (order_days < date_hi_days)

    keys = tl.load(o_orderkey_ptr + offs, mask=date_keep, other=-1)
    key_ok = date_keep & (keys >= 0) & (keys < flag_n)
    has_late_line = tl.load(flags_ptr + keys, mask=key_ok, other=0) != 0

    valid = key_ok & has_late_line
    prio = tl.load(o_orderpriority_codes_ptr + offs, mask=valid, other=-1)

    one_i64 = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    zero_i64 = tl.full((BLOCK_SIZE,), 0, dtype=tl.int64)

    for p in tl.static_range(0, NUM_PRIORITIES):
        c = tl.sum(tl.where(valid & (prio == p), one_i64, zero_i64), axis=0)
        tl.atomic_add(counts_ptr + p, c, sem="relaxed")
