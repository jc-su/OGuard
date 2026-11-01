#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l1_orderkey,
    const int64_t* __restrict__ l1_late_int,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    int64_t* __restrict__ counts,
    int64_t n_elements,
    int64_t n_lo
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) return;
    
    int64_t ok = l1_orderkey[tid];
    if (l1_late_int[tid] == 0) return;
    int64_t sid = supp_ids[tid];
    if (sid < 0) return;
    
    // Optimized binary search with early exit
    int64_t left = 0, right = n_lo - 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
        int64_t mid_val = lo_orderkeys[mid];
        if (mid_val == ok) {
            int64_t distinct = lo_distinct_supp[mid];
            int64_t late_distinct = lo_late_distinct_supp[mid];
            if (distinct >= 2 && late_distinct <= 1) {
                atomicAdd((unsigned long long*)&counts[sid], 1ULL);
            }
            return;
        } else if (mid_val < ok) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
}
