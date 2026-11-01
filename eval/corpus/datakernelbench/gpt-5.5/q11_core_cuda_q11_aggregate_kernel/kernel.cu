#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ partkey_ptr,
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    int64_t n_elements,
    double* __restrict__ dense_values_ptr,
    int32_t* __restrict__ present_ptr,
    double* __restrict__ total_ptr
) {
    __shared__ double block_sum[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    double v = 0.0;
    if (idx < n_elements) {
        const int64_t key = partkey_ptr[idx];
        v = supplycost_ptr[idx] * availqty_ptr[idx];

        if (key >= 0) {
            atomicAdd(dense_values_ptr + key, v);
            present_ptr[key] = 1;
        }
    }

    block_sum[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_sum[tid] += block_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_ptr, block_sum[0]);
    }
}
