#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ flags,
    const double* __restrict__ sums,
    int64_t key_range,
    double threshold,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ big_counts
) {
    __shared__ int u_count[BLOCK_SIZE];
    __shared__ int b_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;
    if (key < key_range && flags[key] != 0) {
        is_unique = 1;
        is_big = (sums[key] > threshold) ? 1 : 0;
    }

    u_count[tid] = is_unique;
    b_count[tid] = is_big;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            u_count[tid] += u_count[tid + offset];
            b_count[tid] += b_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unique_counts[blockIdx.x] = static_cast<int64_t>(u_count[0]);
        big_counts[blockIdx.x] = static_cast<int64_t>(b_count[0]);
    }
}
