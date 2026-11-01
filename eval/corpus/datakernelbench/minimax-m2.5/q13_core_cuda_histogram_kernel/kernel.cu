#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    int64_t n,
    int64_t* __restrict__ counts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        atomicAdd((unsigned long long *)&counts[key], 1ULL);
    }
}
