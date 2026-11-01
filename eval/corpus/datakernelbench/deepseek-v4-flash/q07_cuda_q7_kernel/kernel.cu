#include <cuda_runtime.h>
#define MAX_GROUPS 8

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_nation,
    const int32_t* __restrict__ cust_nation,
    const int32_t* __restrict__ shipdate_days,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue_groups,
    int64_t n_elements
) {
    extern __shared__ double shared_rev[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_vol = 0.0;
    int my_key = -1;
    
    if (idx < n_elements) {
        int32_t sd = shipdate_days[idx];
        if (sd >= date_lo && sd <= date_hi) {
            int32_t supp = supp_nation[idx];
            int32_t cust = cust_nation[idx];
            
            // Compute year from days using simplified algorithm
            int64_t d = (int64_t)sd + 719468;
            int64_t era = (d >= 0 ? d : d - 146096) / 146097;
            int64_t doe = d - era * 146097;
            int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            int64_t y = yoe + era * 400;
            int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            int64_t mp = (5 * doy + 2) / 153;
            int64_t m = mp < 10 ? mp + 3 : mp - 9;
            int64_t year = m <= 2 ? y + 1 : y;
            
            int year_offset = (int)(year - 1995);
            
            if (year_offset >= 0 && year_offset <= 1) {
                my_key = supp * 4 + cust * 2 + year_offset;
                my_vol = price[idx] * (1.0 - discount[idx]);
            }
        }
    }
    
    // Initialize shared memory for this block's group accumulators
    if (threadIdx.x < MAX_GROUPS) {
        shared_rev[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // Accumulate into shared using atomic on shared memory
    if (my_key >= 0 && my_key < MAX_GROUPS) {
        atomicAdd(&shared_rev[my_key], my_vol);
    }
    __syncthreads();
    
    // Final block-level reduction to global
    if (threadIdx.x < MAX_GROUPS) {
        atomicAdd(&revenue_groups[threadIdx.x], shared_rev[threadIdx.x]);
    }
}
