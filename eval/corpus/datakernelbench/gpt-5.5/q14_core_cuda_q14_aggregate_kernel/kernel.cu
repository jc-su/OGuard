#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ sums_ptr,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;

    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if ((sd >= date_lo) && (sd < date_hi)) {
            const double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total = revenue;
            if (promo_ptr[idx]) {
                my_promo = revenue;
            }
        }
    }

    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_total[threadIdx.x] += block_total[threadIdx.x + stride];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sums_ptr + 0, block_total[0]);
        atomicAdd(sums_ptr + 1, block_promo[0]);
    }
}
