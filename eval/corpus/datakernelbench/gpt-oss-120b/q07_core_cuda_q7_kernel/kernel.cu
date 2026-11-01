#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_code,
    const int32_t* __restrict__ cust_code,
    const int32_t* __restrict__ shipdate_days,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const int32_t date_lo,
    const int32_t date_hi,
    double* __restrict__ out_rev,   // 8-element array
    const int64_t n)
{
    // Shared‑memory partial sums for the 8 groups
    __shared__ double block_rev[8];

    // Initialise shared memory (first 8 threads)
    if (threadIdx.x < 8) {
        block_rev[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t sd = shipdate_days[idx];
        // shipdate filter (inclusive)
        if (sd >= date_lo && sd <= date_hi) {
            // ----- year extraction (civil_from_days) -----
            int64_t days = (int64_t)sd + 719468;
            int64_t era = (days >= 0 ? days : days - 146096) / 146097;
            int64_t doe = days - era * 146097;
            int64_t yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
            int64_t y = yoe + era * 400;
            int64_t doy = doe - (365 * yoe + yoe/4 - yoe/100);
            int64_t mp = (5 * doy + 2) / 153;
            int64_t m = (mp < 10) ? mp + 3 : mp - 9;
            int64_t year = (m <= 2) ? y + 1 : y;
            int year_offset = (int)(year - 1995);   // expect 0 or 1
            // ------------------------------------------------------------------
            // Build group key: supp*4 + cust*2 + year_offset  (0..7)
            // ------------------------------------------------------------------
            if (year_offset >= 0 && year_offset < 2) {
                int key = supp_code[idx] * 4 + cust_code[idx] * 2 + year_offset;
                double vol = price[idx] * (1.0 - discount[idx]);
                atomicAdd(&block_rev[key], vol);
            }
        }
    }

    __syncthreads();

    // Reduce per‑block contributions to global result (one atomic per block per key)
    if (threadIdx.x < 8) {
        atomicAdd(&out_rev[threadIdx.x], block_rev[threadIdx.x]);
    }
}
