#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_present,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t sd = l_shipdate[idx];
        if (sd >= date_lo && sd < date_hi) {
            int64_t suppkey = l_suppkey[idx];
            double rev = l_price[idx] * (1.0 - l_discount[idx]);
            // Global atomicAdd. Because selectivity is low (1 quarter out of 7 years = ~3.5%)
            // and spread over up to 100,000 supplier keys, collisions are minimal and 
            // global atomics are very fast without shared memory complexity.
            atomicAdd(&out_revenue[suppkey], rev);
            out_present[suppkey] = 1;
        }
    }
}
