#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_GROUPS 30
#define NUM_AGGS 6

extern "C" __global__ void kernel(
    const int32_t* __restrict__ rf_codes,
    const int32_t* __restrict__ ls_codes,
    const double* __restrict__ quantity,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    double* __restrict__ out_sums,
    int64_t n_elements
) {
    // Shared memory for block-level group totals
    __shared__ double s_final[MAX_GROUPS * NUM_AGGS];
    
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory (thread 0 only)
    if (tid == 0) {
        for (int i = 0; i < MAX_GROUPS * NUM_AGGS; i++) {
            s_final[i] = 0.0;
        }
    }
    __syncthreads();
    
    // Each thread processes its element and accumulates via shared memory atomics
    if (idx < n_elements) {
        if (shipdate_days[idx] <= shipdate_hi_days) {
            int32_t rf = rf_codes[idx];
            int32_t ls = ls_codes[idx];
            // Encode group: rf * 10 + ls (codes are small 0-based integers)
            int32_t g = rf * 10 + ls;
            
            if (g >= 0 && g < MAX_GROUPS) {
                double qty = quantity[idx];
                double p = price[idx];
                double d = discount[idx];
                double t = tax[idx];
                double disc_price = p * (1.0 - d);
                double charge = disc_price * (1.0 + t);
                
                int f_base = g * NUM_AGGS;
                atomicAdd(&s_final[f_base + 0], qty);
                atomicAdd(&s_final[f_base + 1], p);
                atomicAdd(&s_final[f_base + 2], disc_price);
                atomicAdd(&s_final[f_base + 3], charge);
                atomicAdd(&s_final[f_base + 4], d);
                atomicAdd(&s_final[f_base + 5], 1.0);
            }
        }
    }
    __syncthreads();
    
    // Thread 0 writes block totals to global memory
    if (tid == 0) {
        for (int i = 0; i < MAX_GROUPS * NUM_AGGS; i++) {
            if (s_final[i] != 0.0) {
                atomicAdd(&out_sums[i], s_final[i]);
            }
        }
    }
}
