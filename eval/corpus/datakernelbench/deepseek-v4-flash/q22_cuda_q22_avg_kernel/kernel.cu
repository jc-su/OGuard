#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ acctbal,
    const bool* __restrict__ mask,
    double* __restrict__ sum_out,
    long long* __restrict__ count_out,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ long long block_cnt[BLOCK_SIZE];

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    long long my_cnt = 0;

    if (idx < n) {
        bool keep = mask[idx] && (acctbal[idx] > 0.0);
        if (keep) {
            my_sum = acctbal[idx];
            my_cnt = 1;
        }
    }

    block_sum[threadIdx.x] = my_sum;
    block_cnt[threadIdx.x] = my_cnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
            block_cnt[threadIdx.x] += block_cnt[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum[0]);
        atomicAdd((unsigned long long*)count_out, (unsigned long long)block_cnt[0]);
    }
}
