#include <cstdint>

extern "C" __global__ void kernel(const int32_t* group, const int64_t* value,
                                  const uint8_t* valid, int64_t* counts,
                                  int64_t* sums, int64_t n, int64_t g) {
    // Per-block private tables; g is at most 1024. Sums use unsigned 64-bit
    // arithmetic so that an overflow wraps around.
    __shared__ unsigned long long local_counts[1024];
    __shared__ unsigned long long local_sums[1024];
    for (int64_t k = threadIdx.x; k < g; k += 256) {
        local_counts[k] = 0;
        local_sums[k] = 0;
    }
    __syncthreads();
    int64_t stride = (int64_t)gridDim.x * 256;
    for (int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x; i < n; i += stride) {
        if ((valid[i >> 3] >> (i & 7)) & 1) {
            int32_t k = group[i];
            atomicAdd(&local_counts[k], 1ULL);
            atomicAdd(&local_sums[k], (unsigned long long)value[i]);
        }
    }
    __syncthreads();
    for (int64_t k = threadIdx.x; k < g; k += 256) {
        if (local_counts[k] != 0) {
            atomicAdd((unsigned long long*)&counts[k], local_counts[k]);
            atomicAdd((unsigned long long*)&sums[k], local_sums[k]);
        }
    }
}
