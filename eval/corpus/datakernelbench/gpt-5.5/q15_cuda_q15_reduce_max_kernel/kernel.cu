#include <stdint.h>
#include <cfloat>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    double* __restrict__ block_max,
    int64_t n_sup
) {
    __shared__ double smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    double local_max = -DBL_MAX;
    for (int64_t i = idx; i < n_sup; i += stride) {
        if (present[i]) {
            double v = totals[i];
            local_max = v > local_max ? v : local_max;
        }
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
        block_max[blockIdx.x] = smem[0];
    }
}
