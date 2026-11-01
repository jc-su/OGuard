#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ values,
    int64_t n_values,
    int64_t* __restrict__ block_max
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t local_max = 0;

    for (int64_t i = tid; i < n_values; i += stride) {
        int64_t v = values[i];
        if (v > local_max) {
            local_max = v;
        }
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int64_t other = smem[threadIdx.x + s];
            if (other > smem[threadIdx.x]) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}
