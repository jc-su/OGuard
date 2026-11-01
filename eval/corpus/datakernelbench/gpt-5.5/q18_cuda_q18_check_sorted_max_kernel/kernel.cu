#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ max_out,
    int32_t* __restrict__ sorted_flag,
    int64_t n
) {
    __shared__ int64_t smax[BLOCK_SIZE];
    __shared__ int32_t ssorted[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t local_max = 0;
    int32_t local_sorted = 1;

    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        if (k > local_max) local_max = k;
        if (idx > 0) {
            int64_t prev = keys[idx - 1];
            if (k < prev) local_sorted = 0;
        }
    }

    smax[tid] = local_max;
    ssorted[tid] = local_sorted;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
            ssorted[tid] = ssorted[tid] & ssorted[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((unsigned long long*)max_out, (unsigned long long)smax[0]);
        if (ssorted[0] == 0) {
            atomicExch(sorted_flag, 0);
        }
    }
}
