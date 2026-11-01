#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ acctbal_ptr,
    const int32_t* __restrict__ cntrycode_mask_ptr,
    double* __restrict__ sum_out,
    int64_t* __restrict__ count_out,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double my_sum = 0.0;
    int64_t my_count = 0;
    if (idx < n) {
        if (cntrycode_mask_ptr[idx] && acctbal_ptr[idx] > 0.0) {
            my_sum = acctbal_ptr[idx];
            my_count = 1;
        }
    }
    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long*)count_out, (unsigned long long)block_count[0]);
    }
}
