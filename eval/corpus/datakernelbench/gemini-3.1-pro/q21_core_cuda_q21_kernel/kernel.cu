extern "C" __global__ void kernel(
    const int64_t* __restrict__ l1_orderkey,
    const bool* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    unsigned long long int* __restrict__ counts,
    int64_t num_l1,
    int64_t num_lo
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_l1) return;

    // Filter early to avoid expensive binary search when possible
    if (!l1_late[idx]) return;
    int64_t s_id = supp_ids[idx];
    if (s_id < 0) return;

    int64_t okey = l1_orderkey[idx];
    
    // Binary search for the exact orderkey in the unique sorted lo_orderkeys array
    int64_t left = 0;
    int64_t right = num_lo;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (lo_orderkeys[mid] < okey) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Ensure found and retrieve aggregated counts
    if (left < num_lo) {
        int64_t d_supp = lo_distinct_supp[left];
        int64_t l_d_supp = lo_late_distinct_supp[left];
        
        // EXISTS other supplier (distinct_supp >= 2)
        // NOT EXISTS other late supplier (late_distinct_supp <= 1)
        if (d_supp >= 2 && l_d_supp <= 1) {
            atomicAdd(counts + s_id, 1ULL);
        }
    }
}
