#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l1_orderkey,
    const unsigned char* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    unsigned long long* __restrict__ counts,
    const int64_t n,
    const int64_t lo_n)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // ---- load l1 row -------------------------------------------------
    const int64_t okey   = l1_orderkey[idx];
    const unsigned char late = l1_late[idx];
    const int64_t supp_id = supp_ids[idx];

    // ---- binary search (lower_bound) into lo_orderkeys -----------------
    int64_t lo = 0;
    int64_t hi = lo_n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t mid_key = lo_orderkeys[mid];
        if (mid_key < okey)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int64_t pos = lo;   // exact match guaranteed by preprocessing

    // ---- retrieve aggregates for this orderkey ------------------------
    const int64_t distinct = lo_distinct_supp[pos];
    const int64_t late_distinct = lo_late_distinct_supp[pos];

    // ---- evaluate predicates (identical to reference logic) ----------
    const bool exists_other = (distinct >= 2);
    const bool not_exists_other_late = (late_distinct <= 1);
    const bool mask = (late != 0) && exists_other && not_exists_other_late && (supp_id >= 0);

    // ---- atomic increment of per‑supplier counter --------------------
    if (mask) {
        atomicAdd(&counts[supp_id], 1ULL);
    }
}
