#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ block_max,
    int64_t n
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    int64_t v = -9223372036854775807LL - 1LL;

    if (idx < n) {
        v = keys[idx];
    }
    smem[tid] = v;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const int64_t other = smem[tid + s];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}
