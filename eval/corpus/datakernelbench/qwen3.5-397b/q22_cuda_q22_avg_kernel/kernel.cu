#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_mask,
    double* __restrict__ sum_ptr,
    int64_t* __restrict__ count_ptr,
    int64_t n_customers
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_sum = 0.0;
    int64_t my_count = 0;
    
    if (idx < n_customers) {
        if (in_mask[idx] && c_acctbal[idx] > 0.0) {
            my_sum = c_acctbal[idx];
            my_count = 1;
        }
    }
    
    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(sum_ptr, block_sum[0]);
        atomicAdd((unsigned long long*)count_ptr, (unsigned long long)block_count[0]);
    }
}
