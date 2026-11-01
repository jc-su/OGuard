#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ sums,
    double threshold,
    unsigned long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ unsigned int scount[BLOCK_SIZE];

    int tid = threadIdx.x;
    unsigned int local_count = 0;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        if (sums[idx] > threshold) {
            ++local_count;
        }
    }

    scount[tid] = local_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && scount[0] != 0) {
        atomicAdd(count_out, (unsigned long long)scount[0]);
    }
}
