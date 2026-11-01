#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ c_acctbal,
    const bool* __restrict__ in_mask,
    double* __restrict__ avg_sum,
    int64_t* __restrict__ avg_count,
    int64_t n
) {
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ unsigned long long s_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;

    double my_sum = 0.0;
    unsigned long long my_count = 0ULL;

    if (idx < n) {
        double bal = c_acctbal[idx];
        if (in_mask[idx] && bal > 0.0) {
            my_sum = bal;
            my_count = 1ULL;
        }
    }

    s_sum[tid] = my_sum;
    s_count[tid] = my_count;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(avg_sum, s_sum[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(avg_count), s_count[0]);
    }
}
