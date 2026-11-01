#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ total_ptr,
    double* __restrict__ promo_ptr_out,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool in_range = (sd >= date_lo) && (sd < date_hi);
        
        if (in_range) {
            double price = price_ptr[idx];
            double disc = discount_ptr[idx];
            double revenue = price * (1.0 - disc);
            my_total = revenue;
            
            if (promo_ptr[idx]) {
                my_promo = revenue;
            }
        }
    }
    
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();
    
    // Block-level reduction for total
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(total_ptr, block_total[0]);
        atomicAdd(promo_ptr_out, block_promo[0]);
    }
}
