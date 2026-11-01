import triton
import triton.language as tl

TYPES = {
    "l1_orderkey_ptr": "*i64",
    "l1_late_ptr": "*i8",
    "supp_ids_ptr": "*i64",
    "lo_orderkeys_ptr": "*i64",
    "lo_distinct_supp_ptr": "*i64",
    "lo_late_distinct_supp_ptr": "*i64",
    "out_valid_ptr": "*i64",
    "out_count_ptr": "*i64",
    "n_l1": "i32",
    "n_lo": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 1024,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l1_orderkey_ptr,
    l1_late_ptr,
    supp_ids_ptr,
    lo_orderkeys_ptr,
    lo_distinct_supp_ptr,
    lo_late_distinct_supp_ptr,
    out_valid_ptr,     # int64: store valid supp_id (or -1 if not valid)
    out_count_ptr,     # int64: store 1 for valid, 0 for invalid
    n_l1,
    n_lo,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_l1

    l1_ok = tl.load(l1_orderkey_ptr + offs, mask=mask, other=0)
    l1_late_raw = tl.load(l1_late_ptr + offs, mask=mask, other=0)
    l1_late = l1_late_raw.to(tl.int1)
    supp_id = tl.load(supp_ids_ptr + offs, mask=mask, other=-1)

    lo = tl.full((BLOCK_SIZE,), 0, dtype=tl.int64)
    hi = tl.full((BLOCK_SIZE,), n_lo, dtype=tl.int64)
    for _ in tl.static_range(24):
        mid = (lo + hi) // 2
        mid_val = tl.load(lo_orderkeys_ptr + mid, mask=mask, other=0)
        mid_val_lt = mid_val < l1_ok
        lo = tl.where(mid_val_lt & mask, mid + 1, lo)
        hi = tl.where(~mid_val_lt & mask, mid, hi)
    idx = lo
    idx = tl.minimum(idx, n_lo - 1)

    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=mask, other=0)

    exists_other = distinct_supp >= 2
    not_exists_other_late = late_distinct_supp <= 1

    keep = mask & l1_late & exists_other & not_exists_other_late & (supp_id >= 0)

    tl.store(out_valid_ptr + offs, tl.where(keep, supp_id, -1), mask=mask)
    tl.store(out_count_ptr + offs, tl.where(keep, 1, 0), mask=mask)
