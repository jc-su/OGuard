#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024
#define NUM_GROUPS 8

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_ptr,
    const int32_t* __restrict__ cust_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ out_revenue,
    int64_t n_elements
) {
    // Shared memory for per-block group reduction
    __shared__ double block_groups[NUM_GROUPS];

    // Initialize shared memory
    for (int i = threadIdx.x; i < NUM_GROUPS; i += blockDim.x) {
        block_groups[i] = 0.0;
    }
    __syncthreads();

    // Each thread processes one element
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        if (sd >= shipdate_lo_days && sd <= shipdate_hi_days) {
            int32_t supp = supp_ptr[idx];
            int32_t cust = cust_ptr[idx];
            double price = price_ptr[idx];
            double disc = disc_ptr[idx];
            double volume = price * (1.0 - disc);

            // Compute year from days (civil_from_days algorithm)
            int64_t days = (int64_t)sd + 719468;
            int64_t era = days >= 0 ? days : days - 146096;
            era = era / 146097;
            int64_t doe = days - era * 146097;
            int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            int64_t y = yoe + era * 400;
            int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            int64_t mp = (5 * doy + 2) / 153;
            int64_t m = mp < 10 ? mp + 3 : mp - 9;
            int64_t year = m <= 2 ? y + 1 : y;
            int32_t year_offset = (int32_t)year - 1995;

            if (year_offset >= 0 && year_offset <= 1) {
                int32_t group = supp * 4 + cust * 2 + year_offset;
                atomicAdd(&block_groups[group], volume);
            }
        }
    }
    __syncthreads();

    // One thread per group writes to global
    if (threadIdx.x < NUM_GROUPS) {
        atomicAdd(&out_revenue[threadIdx.x], block_groups[threadIdx.x]);
    }
}
