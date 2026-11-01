#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double threshold,
    unsigned long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ unsigned int scount[BLOCK_SIZE];

    int tid = threadIdx.x;
    unsigned int local_count = 0;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        bool is_start = (idx == 0) || (keys[idx - 1] != k);
        if (is_start) {
            double sum = qty[idx];
            int64_t j = idx + 1;
            while (j < n && keys[j] == k) {
                sum += qty[j];
                ++j;
            }
            if (sum > threshold) {
                ++local_count;
            }
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
