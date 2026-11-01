#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_priority,
    const int32_t* __restrict__ o_orderdate,
    int64_t n_orders,
    int64_t date_lo,
    int64_t date_hi,
    const int64_t* __restrict__ late_keys,
    int64_t n_late_keys,
    int64_t* __restrict__ counts,
    int32_t num_priorities
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements
    for (int64_t i = idx; i < n_orders; i += gridDim.x * blockDim.x) {
        int32_t od = o_orderdate[i];
        
        // Filter by date range
        if (od >= (int32_t)date_lo && od < (int32_t)date_hi) {
            int64_t ok = o_orderkey[i];
            
            // Binary search for EXISTS check
            int64_t lo = 0, hi = n_late_keys;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (late_keys[mid] < ok) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            bool found = (lo < n_late_keys && late_keys[lo] == ok);
            
            if (found) {
                int32_t pc = o_priority[i];
                if (pc >= 0 && pc < num_priorities) {
                    atomicAdd((unsigned long long*)&counts[pc], (unsigned long long)1);
                }
            }
        }
    }
}
