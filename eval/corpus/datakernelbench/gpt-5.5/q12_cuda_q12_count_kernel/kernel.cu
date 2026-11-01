#include <cstdint>

#define BLOCK_SIZE 1024

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_code,
    const int32_t* __restrict__ priority_high,
    const int32_t* __restrict__ commit_days,
    const int32_t* __restrict__ receipt_days,
    const int32_t* __restrict__ ship_days,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t* __restrict__ out_counts,
    int64_t n_elements
) {
    __shared__ int64_t sh_high0[BLOCK_SIZE];
    __shared__ int64_t sh_high1[BLOCK_SIZE];
    __shared__ int64_t sh_low0[BLOCK_SIZE];
    __shared__ int64_t sh_low1[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int64_t high0 = 0;
    int64_t high1 = 0;
    int64_t low0 = 0;
    int64_t low1 = 0;

    if (idx < n_elements) {
        const int32_t cd = commit_days[idx];
        const int32_t rd = receipt_days[idx];
        const int32_t sd = ship_days[idx];

        const bool keep = (cd < rd) &&
                          (sd < cd) &&
                          (rd >= receipt_lo_days) &&
                          (rd < receipt_hi_days);

        if (keep) {
            const int32_t mode = shipmode_code[idx];
            const int32_t high = priority_high[idx];

            if (mode == 0) {
                if (high != 0) {
                    high0 = 1;
                } else {
                    low0 = 1;
                }
            } else {
                if (high != 0) {
                    high1 = 1;
                } else {
                    low1 = 1;
                }
            }
        }
    }

    sh_high0[tid] = high0;
    sh_high1[tid] = high1;
    sh_low0[tid] = low0;
    sh_low1[tid] = low1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_high0[tid] += sh_high0[tid + stride];
            sh_high1[tid] += sh_high1[tid + stride];
            sh_low0[tid] += sh_low0[tid + stride];
            sh_low1[tid] += sh_low1[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_i64(out_counts + 0, sh_high0[0]);
        atomic_add_i64(out_counts + 1, sh_high1[0]);
        atomic_add_i64(out_counts + 2, sh_low0[0]);
        atomic_add_i64(out_counts + 3, sh_low1[0]);
    }
}
