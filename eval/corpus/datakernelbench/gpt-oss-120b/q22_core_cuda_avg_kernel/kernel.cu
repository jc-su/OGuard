#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ acctbal,
    const bool*   __restrict__ mask,
    double*       __restrict__ out_sum,
    unsigned long long* __restrict__ out_cnt,
    int64_t n)
{
    double my_sum = 0.0;
    unsigned long long my_cnt = 0ULL;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx] && acctbal[idx] > 0.0) {
            my_sum = acctbal[idx];
            my_cnt = 1ULL;
        }
    }

    // shared‑memory reduction
    extern __shared__ char shm[];
    double* s_sum = (double*)shm;
    unsigned long long* s_cnt = (unsigned long long*)(s_sum + BLOCK_SIZE);
    s_sum[threadIdx.x] = my_sum;
    s_cnt[threadIdx.x] = my_cnt;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_sum, s_sum[0]);
        atomicAdd(out_cnt, s_cnt[0]);
    }
}
