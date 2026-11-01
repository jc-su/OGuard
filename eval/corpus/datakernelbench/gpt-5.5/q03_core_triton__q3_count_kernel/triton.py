import triton
import triton.language as tl

TYPES = {
    "l_shipdate_days_ptr": "*i32",
    "o_orderdate_days_ptr": "*i32",
    "keep_ptr": "*u8",
    "counts_ptr": "*i64",
    "shipdate_lo_days": "i32",
    "orderdate_hi_days": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_shipdate_days_ptr,
    o_orderdate_days_ptr,
    keep_ptr,
    counts_ptr,
    shipdate_lo_days,
    orderdate_hi_days,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_shipdate = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    o_orderdate = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=orderdate_hi_days)

    keep = mask & (l_shipdate > shipdate_lo_days) & (o_orderdate < orderdate_hi_days)
    tl.store(keep_ptr + offs, keep.to(tl.uint8), mask=mask)

    cnt = tl.sum(keep.to(tl.int64), axis=0)
    tl.store(counts_ptr + pid, cnt)
