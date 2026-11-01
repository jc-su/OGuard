#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ total_out,
    double* __restrict__ promo_out,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];

    double my_total = 0.0;
    double my_promo = 0.0;

    // Only process valid indices
    for (int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < n_elements; idx += gridDim.x * BLOCK_SIZE) {
        int32_t sd = shipdate_ptr[idx];
        if (sd >= date_lo && sd < date_hi) {
            double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total += rev;
            if (promo_ptr[idx]) {
                my_promo += rev;
            }
        }
    }

    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();

    // Tree reduction in shared memory (fully safe)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total_out, block_total[0]);
        atomicAdd(promo_out, block_promo[0]);
    }
}
