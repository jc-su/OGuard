#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    int64_t n,
    unsigned long long* __restrict__ max_out
) {
    __shared__ unsigned long long smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    unsigned long long v = 0ULL;

    if (idx < n) {
        int64_t k = keys[idx];
        v = (k > 0) ? static_cast<unsigned long long>(k) : 0ULL;
    }

    smem[tid] = v;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            unsigned long long other = smem[tid + offset];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_out, smem[0]);
    }
}
