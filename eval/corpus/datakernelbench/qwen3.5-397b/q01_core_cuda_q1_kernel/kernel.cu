#include <cuda_runtime.h>

#define MAX_GROUPS 1024
#define NUM_AGGS 6

extern "C" __global__ void kernel(
    const int32_t* rf,
    const int32_t* ls,
    const double* qty,
    const double* price,
    const double* disc,
    const double* tax,
    const int32_t* shipdate,
    int32_t shipdate_hi,
    double* accum,
    int64_t n
) {
    extern __shared__ double shmem[];
    // Initialize shared memory
    for (int i = threadIdx.x; i < MAX_GROUPS * NUM_AGGS; i += blockDim.x) {
        shmem[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int64_t i = idx; i < n; i += stride) {
        if (shipdate[i] <= shipdate_hi) {
            int32_t r = rf[i];
            int32_t l = ls[i];
            int32_t key = r * 256 + l;
            if (key < MAX_GROUPS) {
                double q = qty[i];
                double p = price[i];
                double d = disc[i];
                double t = tax[i];
                double dp = p * (1.0 - d);
                double ch = dp * (1.0 + t);
                int base = key * NUM_AGGS;
                // Accumulate in shared memory
                atomicAdd(&shmem[base + 0], q);
                atomicAdd(&shmem[base + 1], p);
                atomicAdd(&shmem[base + 2], dp);
                atomicAdd(&shmem[base + 3], ch);
                atomicAdd(&shmem[base + 4], d);
                atomicAdd(&shmem[base + 5], 1.0);
            }
        }
    }
    __syncthreads();

    // Write back to global memory (only thread 0 per block)
    if (threadIdx.x == 0) {
        for (int k = 0; k < MAX_GROUPS; k++) {
            int base = k * NUM_AGGS;
            double count = shmem[base + 5];
            if (count > 0.0) {
                atomicAdd(&accum[base + 0], shmem[base + 0]);
                atomicAdd(&accum[base + 1], shmem[base + 1]);
                atomicAdd(&accum[base + 2], shmem[base + 2]);
                atomicAdd(&accum[base + 3], shmem[base + 3]);
                atomicAdd(&accum[base + 4], shmem[base + 4]);
                atomicAdd(&accum[base + 5], count);
            }
        }
    }
}
