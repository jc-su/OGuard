#include <cstdint>

#define BLOCK_SIZE 256
#define NUM_GROUPS 8

__device__ __forceinline__ int32_t year_from_epoch_days(int32_t days_in) {
    // Howard Hinnant civil_from_days, matching the reference arithmetic for positive TPC-H dates.
    int64_t z = static_cast<int64_t>(days_in) + 719468LL;
    int64_t era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    int64_t doe = z - era * 146097LL;
    int64_t yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    int64_t y = yoe + era * 400LL;
    int64_t doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    int64_t mp = (5LL * doy + 2LL) / 153LL;
    int64_t m = (mp < 10LL) ? (mp + 3LL) : (mp - 9LL);
    int64_t year = (m <= 2LL) ? (y + 1LL) : y;
    return static_cast<int32_t>(year);
}

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_ptr,
    const int32_t* __restrict__ cust_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue8_ptr,
    int64_t n_elements
) {
    __shared__ double block_vals[NUM_GROUPS * BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; ++g) {
        block_vals[g * BLOCK_SIZE + tid] = 0.0;
    }

    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if (sd >= date_lo && sd <= date_hi) {
            const int32_t supp = supp_ptr[idx];
            const int32_t cust = cust_ptr[idx];
            const int32_t year = year_from_epoch_days(sd);
            const int32_t year_offset = year - 1995;
            const int64_t key = static_cast<int64_t>(supp) * 4LL
                              + static_cast<int64_t>(cust) * 2LL
                              + static_cast<int64_t>(year_offset);
            if (key >= 0 && key < NUM_GROUPS) {
                const double price = price_ptr[idx];
                const double disc = discount_ptr[idx];
                block_vals[key * BLOCK_SIZE + tid] = price * (1.0 - disc);
            }
        }
    }

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int g = 0; g < NUM_GROUPS; ++g) {
                block_vals[g * BLOCK_SIZE + tid] += block_vals[g * BLOCK_SIZE + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < NUM_GROUPS) {
        atomicAdd(revenue8_ptr + tid, block_vals[tid * BLOCK_SIZE]);
    }
}
