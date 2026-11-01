#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ flags,
    const double* __restrict__ sums,
    int64_t key_range,
    double threshold,
    const int64_t* __restrict__ unique_offsets,
    const int64_t* __restrict__ big_offsets,
    int64_t* __restrict__ unique_out,
    int64_t* __restrict__ big_keys_out,
    double* __restrict__ big_sums_out
) {
    __shared__ int u_scan[BLOCK_SIZE];
    __shared__ int b_scan[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;
    double s = 0.0;

    if (key < key_range && flags[key] != 0) {
        s = sums[key];
        is_unique = 1;
        is_big = (s > threshold) ? 1 : 0;
    }

    u_scan[tid] = is_unique;
    b_scan[tid] = is_big;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add_u = (tid >= offset) ? u_scan[tid - offset] : 0;
        int add_b = (tid >= offset) ? b_scan[tid - offset] : 0;
        __syncthreads();
        u_scan[tid] += add_u;
        b_scan[tid] += add_b;
        __syncthreads();
    }

    if (is_unique) {
        int64_t pos = unique_offsets[blockIdx.x] + static_cast<int64_t>(u_scan[tid] - 1);
        unique_out[pos] = key;
    }
    if (is_big) {
        int64_t pos = big_offsets[blockIdx.x] + static_cast<int64_t>(b_scan[tid] - 1);
        big_keys_out[pos] = key;
        big_sums_out[pos] = s;
    }
}
