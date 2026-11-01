#include <cstdint>

#define BLOCK_SIZE 1024

__device__ __forceinline__ void atomic_add_i64_nonnegative(
    int64_t* addr,
    long long value
) {
    if (value != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(addr),
            static_cast<unsigned long long>(value)
        );
    }
}

extern "C" __global__ void kernel(
    const int32_t* __restrict__ l_shipmode_code,
    const int32_t* __restrict__ o_orderpriority_high,
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_shipdate,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t n_elements,
    int64_t* __restrict__ high_counts,
    int64_t* __restrict__ low_counts
) {
    __shared__ long long shared_counts[4 * BLOCK_SIZE];

    long long* sh_h0 = shared_counts;
    long long* sh_h1 = shared_counts + BLOCK_SIZE;
    long long* sh_l0 = shared_counts + 2 * BLOCK_SIZE;
    long long* sh_l1 = shared_counts + 3 * BLOCK_SIZE;

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    long long h0 = 0;
    long long h1 = 0;
    long long l0 = 0;
    long long l1 = 0;

    if (idx < n_elements) {
        const int32_t commit_d = l_commitdate[idx];
        const int32_t receipt_d = l_receiptdate[idx];
        const int32_t ship_d = l_shipdate[idx];

        const bool keep =
            (commit_d < receipt_d) &&
            (ship_d < commit_d) &&
            (receipt_d >= receipt_lo_days) &&
            (receipt_d < receipt_hi_days);

        if (keep) {
            const int32_t code = l_shipmode_code[idx];
            const int32_t high = o_orderpriority_high[idx] != 0 ? 1 : 0;
            const int32_t low = 1 - high;

            if (code == 0) {
                h0 = static_cast<long long>(high);
                l0 = static_cast<long long>(low);
            } else {
                h1 = static_cast<long long>(high);
                l1 = static_cast<long long>(low);
            }
        }
    }

    sh_h0[tid] = h0;
    sh_h1[tid] = h1;
    sh_l0[tid] = l0;
    sh_l1[tid] = l1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_h0[tid] += sh_h0[tid + stride];
            sh_h1[tid] += sh_h1[tid + stride];
            sh_l0[tid] += sh_l0[tid + stride];
            sh_l1[tid] += sh_l1[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_i64_nonnegative(&high_counts[0], sh_h0[0]);
        atomic_add_i64_nonnegative(&high_counts[1], sh_h1[0]);
        atomic_add_i64_nonnegative(&low_counts[0], sh_l0[0]);
        atomic_add_i64_nonnegative(&low_counts[1], sh_l1[0]);
    }
}
