#include <cstdint>

#define BLOCK_SIZE 1024

__device__ __forceinline__ int find_code_idx(
    int32_t code,
    const int32_t* __restrict__ codes,
    int num_codes
) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= num_codes) break;
        if (codes[i] == code) return i;
    }
    for (int i = 16; i < num_codes; ++i) {
        if (codes[i] == code) return i;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const int32_t* __restrict__ codes,
    int num_codes,
    double* __restrict__ avg_stats,
    int64_t n_customers
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ double block_count[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    double my_count = 0.0;

    if (idx < n_customers) {
        double bal = c_acctbal[idx];
        if (bal > 0.0) {
            int32_t code = cntrycode[idx];
            if (find_code_idx(code, codes, num_codes) >= 0) {
                my_sum = bal;
                my_count = 1.0;
            }
        }
    }

    block_sum[threadIdx.x] = my_sum;
    block_count[threadIdx.x] = my_count;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
            block_count[threadIdx.x] += block_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&avg_stats[0], block_sum[0]);
        atomicAdd(&avg_stats[1], block_count[0]);
    }
}
