#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ dense_values_ptr,
    const int32_t* __restrict__ present_ptr,
    int64_t dense_len,
    double threshold,
    unsigned long long* __restrict__ count_ptr
) {
    __shared__ unsigned int block_count[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    unsigned int keep = 0;
    if (idx < dense_len) {
        keep = (present_ptr[idx] != 0 && dense_values_ptr[idx] > threshold) ? 1u : 0u;
    }

    block_count[tid] = keep;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_count[tid] += block_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && block_count[0] != 0u) {
        atomicAdd(count_ptr, static_cast<unsigned long long>(block_count[0]));
    }
}
