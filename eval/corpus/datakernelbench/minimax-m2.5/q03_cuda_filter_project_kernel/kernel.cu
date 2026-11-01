#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_shippriority,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    int64_t* __restrict__ out_orderkey,
    int32_t* __restrict__ out_orderdate,
    int32_t* __restrict__ out_shippriority,
    double* __restrict__ out_revenue,
    unsigned long long* __restrict__ out_write_idx,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        int32_t lsd = l_shipdate_days[idx];
        int32_t ood = o_orderdate_days[idx];
        
        if (lsd > shipdate_lo_days && ood < orderdate_hi_days) {
            // Compute revenue
            double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            
            // Get write position atomically
            unsigned long long write_pos = atomicAdd(out_write_idx, 1ULL);
            
            out_orderkey[write_pos] = l_orderkey[idx];
            out_orderdate[write_pos] = ood;
            out_shippriority[write_pos] = o_shippriority[idx];
            out_revenue[write_pos] = rev;
        }
    }
}
