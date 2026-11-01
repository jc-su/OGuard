#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_cntrycode_mask,
    int64_t n_customers,
    double* sum_out,
    int64_t* count_out
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ int64_t block_count[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    double my_sum = 0.0;
    int64_t my_count = 0;
    
    if (idx < n_customers) {
        if (in_cntrycode_mask[idx] && c_acctbal[idx] > 0.0) {
            my_sum = c_acctbal[idx];
            my_count = 1;
        }
    }
    
    block_sum[tid] = my_sum;
    block_count[tid] = my_count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
            block_count[tid] += block_count[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long int*)count_out, (unsigned long long int)block_count[0]);
    }
}
