#include <stdint.h>
#include <cfloat>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ block_max,
    double* __restrict__ max_out,
    int64_t n_blocks
) {
    __shared__ double smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    double local_max = -DBL_MAX;
    for (int64_t i = tid; i < n_blocks; i += blockDim.x) {
        double v = block_max[i];
        local_max = v > local_max ? v : local_max;
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other = smem[tid + s];
            smem[tid] = other > smem[tid] ? other : smem[tid];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_out[0] = smem[0];
    }
}
