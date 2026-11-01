#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ block_max,
    int64_t* __restrict__ out,
    int64_t n_blocks
) {
    __shared__ int64_t smem[BLOCK_SIZE];

    const int tid = threadIdx.x;
    int64_t v = -9223372036854775807LL - 1LL;

    for (int64_t i = tid; i < n_blocks; i += blockDim.x) {
        const int64_t x = block_max[i];
        if (x > v) {
            v = x;
        }
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
        out[0] = smem[0];
    }
}
