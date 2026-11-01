import triton
import triton.language as tl

TYPES = {
    "l1_orderkey_ptr": "*i64",
    "l1_late_ptr": "*i1",
    "supp_ids_ptr": "*i64",
    "lo_orderkeys_ptr": "*i64",
    "lo_distinct_supp_ptr": "*i64",
    "lo_late_distinct_supp_ptr": "*i64",
    "counts_ptr": "*i64",
    "n_l1": "i32",
    "lo_n": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256, "SEARCH_ITERS": 23}

NUM_WARPS = 4


@triton.jit
def kernel(
    l1_orderkey_ptr,
    l1_late_ptr,
    supp_ids_ptr,
    lo_orderkeys_ptr,
    lo_distinct_supp_ptr,
    lo_late_distinct_supp_ptr,
    counts_ptr,
    n_l1,
    lo_n,
    BLOCK_SIZE: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_l1

    late = tl.load(l1_late_ptr + offs, mask=row_mask, other=0)
    supp_id = tl.load(supp_ids_ptr + offs, mask=row_mask, other=-1)

    # Filter early so non-late rows do not issue random binary-search loads.
    cand = row_mask & (late != 0) & (supp_id >= 0)
    key = tl.load(l1_orderkey_ptr + offs, mask=cand, other=0)

    low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    high = low + lo_n

    # Vectorized lower_bound(searchsorted) over sorted unique lo_orderkeys.
    for _ in tl.static_range(0, SEARCH_ITERS):
        mid = (low + high) // 2
        mid_valid = cand & (mid < lo_n)
        mid_key = tl.load(
            lo_orderkeys_ptr + mid,
            mask=mid_valid,
            other=9223372036854775807,
        )
        go_right = mid_key < key
        low = tl.where(cand & go_right, mid + 1, low)
        high = tl.where(cand & (~go_right), mid, high)

    idx = low
    stat_mask = cand & (idx < lo_n)
    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=stat_mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=stat_mask, other=0)

    keep = stat_mask & (distinct_supp >= 2) & (late_distinct_supp <= 1)
    safe_supp_id = tl.where(keep, supp_id, 0)
    one = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    tl.atomic_add(counts_ptr + safe_supp_id, one, sem="relaxed", mask=keep)
