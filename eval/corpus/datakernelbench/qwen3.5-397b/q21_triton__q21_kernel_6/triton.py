import triton
import triton.language as tl

TYPES = {
    "l1_orderkey_ptr": "*i64",
    "l1_supp_id_ptr": "*i32",
    "agg_orderkey_ptr": "*i64",
    "agg_distinct_ptr": "*i32",
    "agg_late_distinct_ptr": "*i32",
    "counts_ptr": "*i64",
    "n_l1": "i32",
    "n_agg": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    l1_orderkey_ptr,
    l1_supp_id_ptr,
    agg_orderkey_ptr,
    agg_distinct_ptr,
    agg_late_distinct_ptr,
    counts_ptr,
    n_l1,
    n_agg,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_l1

    # Load l1 data
    l1_ok = tl.load(l1_orderkey_ptr + offs, mask=mask, other=0)
    l1_sid = tl.load(l1_supp_id_ptr + offs, mask=mask, other=0)

    # Binary Search for each element in the block to find index in agg_orderkey
    # Since agg_orderkey is sorted, we find the position where l1_ok == agg_orderkey
    # We implement a scalar binary search per lane.
    # To avoid dynamic loop divergence overhead, we use a fixed iteration count sufficient for max orders (~15M -> 24 iters)
    # However, standard while loop is supported in Triton.
    
    # We need to compute idx for each lane.
    # Initialize low/high
    low = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    high = tl.full([BLOCK_SIZE], n_agg, dtype=tl.int32)
    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # 25 iterations covers up to 33M elements (2^25)
    for _ in tl.static_range(25):
        mid = (low + high) // 2
        # Prevent OOB load if mid >= n_agg (should not happen if high starts at n_agg and logic is correct, 
        # but mid can be n_agg if low=n_agg, high=n_agg+1? No high starts at n_agg.
        # If low=n_agg-1, high=n_agg, mid=n_agg-1.
        # If low=n_agg, high=n_agg, loop condition low<high fails.
        # So mid is always < n_agg inside loop if low < high.
        # But tl.load needs a pointer. We clamp mid to n_agg-1 for safety if logic drifts, 
        # but strict binary search logic ensures mid < high <= n_agg.
        # Wait, if high=n_agg, mid can be n_agg-1.
        # We must ensure we don't load out of bounds.
        # Safe load:
        val = tl.load(agg_orderkey_ptr + mid, mask=mid < n_agg, other=0)
        
        # Update bounds
        # if val < target: low = mid + 1
        # else: high = mid
        # Use tl.where for vectorization
        cond = val < l1_ok
        low = tl.where(cond, mid + 1, low)
        high = tl.where(cond, high, mid)

    idx = low # This is the searchsorted result (left insertion point)

    # Check if found (idx < n_agg and agg_orderkey[idx] == l1_ok)
    # In our logic, l1_ok must exist in agg.
    found_mask = mask & (idx < n_agg)
    
    # Load aggregate stats
    agg_dist = tl.load(agg_distinct_ptr + idx, mask=found_mask, other=0)
    agg_late_dist = tl.load(agg_late_distinct_ptr + idx, mask=found_mask, other=0)

    # Conditions:
    # 1. distinct_supp >= 2 (exists other supplier)
    # 2. late_distinct_supp <= 1 (no other late supplier, since l1 is late)
    # Note: l1 is already filtered for late in host, so we only check order stats.
    cond1 = agg_dist >= 2
    cond2 = agg_late_dist <= 1
    
    do_count = found_mask & cond1 & cond2

    # Atomic add to counts
    # l1_sid is int32 index into counts
    tl.atomic_add(counts_ptr + l1_sid, 1, mask=do_count)
