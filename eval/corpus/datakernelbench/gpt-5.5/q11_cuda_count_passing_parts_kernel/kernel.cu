#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ part_sums,
    const double* __restrict__ total_sum,
    double threshold_fraction,
    int64_t part_sums_len,
    int64_t* __restrict__ out_count
) {
    __shared__ int block_count[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double threshold = total_sum[0] * threshold_fraction;
    int local = 0;

    if (idx < part_sums_len) {
        double v = part_sums[idx];
        local = (v > threshold) ? 1 : 0;
    }

    block_count[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_count[threadIdx.x] += block_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && block_count[0] != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(out_count),
                  static_cast<unsigned long long>(block_count[0]));
    }
}
