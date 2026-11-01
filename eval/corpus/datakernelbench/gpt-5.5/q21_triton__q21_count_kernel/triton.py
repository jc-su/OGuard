import triton
import triton.language as tl

TYPES = {
    "cand_orderkey_ptr": "*i64",
    "cand_supp_id_ptr": "*i64",
    "cand_late_count_ptr": "*i64",
    "stat_orderkey_ptr": "*i64",
    "stat_distinct_ptr": "*i64",
    "stat_late_distinct_ptr": "*i64",
    "counts_ptr": "*i64",
    "n_cand": "i32",
    "n_stats": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 8


@triton.jit
def kernel(
    cand_orderkey_ptr,
    cand_supp_id_ptr,
    cand_late_count_ptr,
    stat_orderkey_ptr,
    stat_distinct_ptr,
    stat_late_distinct_ptr,
    counts_ptr,
    n_cand,
    n_stats,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cand

    key = tl.load(cand_orderkey_ptr + offs, mask=mask, other=0)
    supp_id = tl.load(cand_supp_id_ptr + offs, mask=mask, other=0)
    late_count = tl.load(cand_late_count_ptr + offs, mask=mask, other=0)

    # Vectorized lower_bound over sorted per-order statistics.
    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    hi = lo + n_stats
    hi = tl.where(mask, hi, lo)

    for _ in tl.static_range(0, 32):
        active = lo < hi
        mid = (lo + hi) // 2
        mid_key = tl.load(stat_orderkey_ptr + mid, mask=active, other=9223372036854775807)
        go_right = mid_key < key
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)

    in_range = mask & (lo < n_stats)
    found_key = tl.load(stat_orderkey_ptr + lo, mask=in_range, other=-1)
    found = in_range & (found_key == key)

    distinct_supp = tl.load(stat_distinct_ptr + lo, mask=found, other=0)
    late_distinct_supp = tl.load(stat_late_distinct_ptr + lo, mask=found, other=2)

    keep = found & (late_count > 0) & (distinct_supp >= 2) & (late_distinct_supp <= 1)
    tl.atomic_add(counts_ptr + supp_id, late_count, sem="relaxed", mask=keep)
