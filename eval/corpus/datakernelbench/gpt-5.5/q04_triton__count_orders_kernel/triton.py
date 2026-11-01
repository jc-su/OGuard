import triton
import triton.language as tl

TYPES = {
    "o_orderkey_ptr": "*i64",
    "o_orderdate_days_ptr": "*i32",
    "o_priority_codes_ptr": "*i32",
    "bitmap_ptr": "*u8",
    "counts_ptr": "*i64",
    "n_elements": "i32",
    "bitmap_size": "i32",
    "date_lo_days": "i32",
    "date_hi_days": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024, "MAX_PRIORITIES": 5}

NUM_WARPS = 8


@triton.jit
def kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_priority_codes_ptr,
    bitmap_ptr,
    counts_ptr,
    n_elements,
    bitmap_size,
    date_lo_days,
    date_hi_days,
    BLOCK_SIZE: tl.constexpr,
    MAX_PRIORITIES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderdate = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=date_hi_days)
    date_keep = mask & (orderdate >= date_lo_days) & (orderdate < date_hi_days)

    orderkey = tl.load(o_orderkey_ptr + offs, mask=date_keep, other=-1)
    key_ok = date_keep & (orderkey >= 0) & (orderkey < bitmap_size)

    late_flag = tl.load(bitmap_ptr + orderkey, mask=key_ok, other=0)
    exists_keep = key_ok & (late_flag != 0)

    prio = tl.load(o_priority_codes_ptr + offs, mask=exists_keep, other=-1)

    for g in tl.static_range(0, MAX_PRIORITIES):
        inc = tl.where(exists_keep & (prio == g), 1, 0)
        cnt = tl.sum(inc, axis=0).to(tl.int64)
        tl.atomic_add(counts_ptr + g, cnt, sem="relaxed")
