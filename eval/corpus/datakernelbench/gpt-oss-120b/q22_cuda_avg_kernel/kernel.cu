#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

extern "C" __global__ void kernel(
    const double* __restrict__ bal,
    const bool*   __restrict__ mask,
    double*       __restrict__ sum_out,
    unsigned long long* __restrict__ cnt_out,
    int64_t N)
{
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ unsigned long long s_cnt[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;

    double my_sum = 0.0;
    unsigned long long my_cnt = 0ULL;

    if (idx < N) {
        bool keep = mask[idx] && bal[idx] > 0.0;
        if (keep) {
            my_sum = bal[idx];
            my_cnt = 1ULL;
        }
    }
    s_sum[tid] = my_sum;
    s_cnt[tid] = my_cnt;
    __syncthreads();

    // reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        __syncthreads();
    }

    // block‑level atomics
    if (tid == 0) {
        atomicAdd(sum_out, s_sum[0]);
        atomicAdd(cnt_out, s_cnt[0]);
    }
}
