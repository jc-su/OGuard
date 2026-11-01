import triton
import triton.language as tl

TYPES = {
    "l_suppkey_ptr": "*i64",
    "l_price_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_shipdate_days_ptr": "*i32",
    "sorted_suppkey_ptr": "*i64",
    "totals_ptr": "*fp64",
    "present_ptr": "*u8",
    "shipdate_lo_days": "i32",
    "shipdate_hi_days": "i32",
}

CONSTANTS = {"n_line": 59986052, "n_sup": 100000, "BINARY_STEPS": 18, "BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_suppkey_ptr,
    l_price_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    sorted_suppkey_ptr,
    totals_ptr,
    present_ptr,
    shipdate_lo_days,
    shipdate_hi_days,
    n_line: tl.constexpr,
    n_sup: tl.constexpr,
    BINARY_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_line

    shipdate = tl.load(l_shipdate_days_ptr + offs, mask=row_mask, other=shipdate_hi_days)
    date_keep = row_mask & (shipdate >= shipdate_lo_days) & (shipdate < shipdate_hi_days)

    supp = tl.load(l_suppkey_ptr + offs, mask=date_keep, other=0)

    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    hi = lo + n_sup

    for _ in tl.static_range(0, BINARY_STEPS):
        active = date_keep & (lo < hi)
        mid = (lo + hi) // 2
        mid_val = tl.load(sorted_suppkey_ptr + mid, mask=active, other=0)
        go_right = mid_val < supp
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & (~go_right), mid, hi)

    pos = lo
    pos_in = date_keep & (pos < n_sup)
    found_key = tl.load(sorted_suppkey_ptr + pos, mask=pos_in, other=0)
    found = pos_in & (found_key == supp)

    price = tl.load(l_price_ptr + offs, mask=found, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=found, other=0.0)
    revenue = price * (1.0 - discount)

    tl.atomic_add(totals_ptr + pos, revenue, sem="relaxed", mask=found)
    tl.store(present_ptr + pos, 1, mask=found)
