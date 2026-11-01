#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode,
    const int32_t* __restrict__ priority,
    const int32_t* __restrict__ commit,
    const int32_t* __restrict__ receipt,
    const int32_t* __restrict__ ship,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ high_out,
    unsigned long long* __restrict__ low_out,
    const int64_t N)
{
    // per‑thread tallies (keep as 0/1)
    unsigned long long h0 = 0ULL, h1 = 0ULL;
    unsigned long long l0 = 0ULL, l1 = 0ULL;

    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < N) {
        int32_t c = commit[idx];
        int32_t r = receipt[idx];
        int32_t s = ship[idx];
        bool keep = (c < r) && (s < c) && (r >= receipt_lo) && (r < receipt_hi);
        if (keep) {
            int32_t code = shipmode[idx];      // 0 or 1
            int32_t high = priority[idx];      // 1 for high, 0 otherwise
            int32_t low  = 1 - high;
            if (code == 0) {
                h0 = (unsigned long long)high;
                l0 = (unsigned long long)low;
            } else {
                h1 = (unsigned long long)high;
                l1 = (unsigned long long)low;
            }
        }
    }

    // Shared memory buffers for reduction (one per group)
    __shared__ unsigned long long sh_h0[BLOCK_SIZE];
    __shared__ unsigned long long sh_h1[BLOCK_SIZE];
    __shared__ unsigned long long sh_l0[BLOCK_SIZE];
    __shared__ unsigned long long sh_l1[BLOCK_SIZE];

    sh_h0[threadIdx.x] = h0;
    sh_h1[threadIdx.x] = h1;
    sh_l0[threadIdx.x] = l0;
    sh_l1[threadIdx.x] = l1;
    __syncthreads();

    // Tree reduction inside the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_h0[threadIdx.x] += sh_h0[threadIdx.x + stride];
            sh_h1[threadIdx.x] += sh_h1[threadIdx.x + stride];
            sh_l0[threadIdx.x] += sh_l0[threadIdx.x + stride];
            sh_l1[threadIdx.x] += sh_l1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One atomic add per group per block
    if (threadIdx.x == 0) {
        atomicAdd(high_out + 0, sh_h0[0]);
        atomicAdd(high_out + 1, sh_h1[0]);
        atomicAdd(low_out  + 0, sh_l0[0]);
        atomicAdd(low_out  + 1, sh_l1[0]);
    }
}
