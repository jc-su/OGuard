#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    double* __restrict__ groups,
    double* __restrict__ total,
    int64_t n)
{
    // shared memory reduction within a block
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < n) {
        my_val = supplycost[idx] * availqty[idx];
        // atomic addition into the per‑key bucket
        atomicAdd(&groups[keys[idx]], my_val);
    }
    block_sum[threadIdx.x] = my_val;
    __syncthreads();

    // intra‑block reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomicAdd per block to the global total
    if (threadIdx.x == 0) {
        atomicAdd(total, block_sum[0]);
    }
}
