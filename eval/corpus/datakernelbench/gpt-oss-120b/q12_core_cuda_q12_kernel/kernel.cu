#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_code,   // 0 = MAIL, 1 = SHIP
    const int32_t* __restrict__ priority_high,   // 1 = high priority, 0 = low
    const int32_t* __restrict__ commitdate,
    const int32_t* __restrict__ receiptdate,
    const int32_t* __restrict__ shipdate,
    const int32_t receipt_lo,
    const int32_t receipt_hi,
    unsigned long long* __restrict__ high_counts,   // length 2
    unsigned long long* __restrict__ low_counts,    // length 2
    const int64_t n_elements)
{
    // per‑block shared accumulators
    __shared__ unsigned long long block_high[2];
    __shared__ unsigned long long block_low[2];

    // initialise shared memory (only first two threads needed)
    if (threadIdx.x < 2) {
        block_high[threadIdx.x] = 0ULL;
        block_low[threadIdx.x]  = 0ULL;
    }
    __syncthreads();

    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        const int32_t cd = commitdate[idx];
        const int32_t rd = receiptdate[idx];
        const int32_t sd = shipdate[idx];
        bool keep = (cd < rd) &&
                    (sd < cd) &&
                    (rd >= receipt_lo) &&
                    (rd < receipt_hi);
        if (keep) {
            const int32_t mode = shipmode_code[idx];   // 0 or 1
            const int32_t high = priority_high[idx];   // 0 or 1
            const unsigned long long high_val = static_cast<unsigned long long>(high);
            const unsigned long long low_val  = static_cast<unsigned long long>(1 - high);
            // block‑level atomics (fast, low contention)
            atomicAdd(&block_high[mode], high_val);
            atomicAdd(&block_low[mode],  low_val);
        }
    }
    __syncthreads();

    // one thread per group adds the block's contribution to global counters
    if (threadIdx.x < 2) {
        atomicAdd(&high_counts[threadIdx.x], block_high[threadIdx.x]);
        atomicAdd(&low_counts[threadIdx.x],  block_low[threadIdx.x]);
    }
}
