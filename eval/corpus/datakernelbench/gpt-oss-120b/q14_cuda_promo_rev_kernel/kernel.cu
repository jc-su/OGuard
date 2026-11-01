#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__device__ __forceinline__ void block_reduce(double *shared_tot,
                                            double *shared_promo,
                                            int tid) {
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_tot[tid]   += shared_tot[tid + stride];
            shared_promo[tid] += shared_promo[tid + stride];
        }
        __syncthreads();
    }
}

extern "C" __global__ void kernel(
    const double *__restrict__ extprice,
    const double *__restrict__ discount,
    const int32_t *__restrict__ shipdate,
    const uint8_t *__restrict__ promo,
    int32_t date_lo,
    int32_t date_hi,
    double *__restrict__ out_total,
    double *__restrict__ out_promo,
    int64_t N) {

    double my_total = 0.0;
    double my_promo = 0.0;

    int64_t idx = blockIdx.x * (int64_t)BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        int32_t sd = shipdate[idx];
        // shipdate filter (already pre‑filtered, but kept for correctness)
        if (sd >= date_lo && sd < date_hi) {
            double rev = extprice[idx] * (1.0 - discount[idx]);
            my_total = rev;
            if (promo[idx]) {
                my_promo = rev;
            }
        }
    }

    // shared memory for block reduction
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;

    // intra‑block reduction
    block_reduce(block_total, block_promo, threadIdx.x);

    // atomic add once per block
    if (threadIdx.x == 0) {
        atomicAdd(out_total, block_total[0]);
        atomicAdd(out_promo, block_promo[0]);
    }
}
