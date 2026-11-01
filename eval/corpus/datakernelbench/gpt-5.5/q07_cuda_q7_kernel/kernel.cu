#include <cstdint>

#define BLOCK_SIZE 1024
#define NUM_GROUPS 8

__device__ __forceinline__ int civil_year_from_days(int32_t z_in) {
    // Howard Hinnant civil calendar conversion. Input is days since 1970-01-01.
    int64_t z = static_cast<int64_t>(z_in) + 719468LL;
    int64_t era = (z >= 0 ? z : z - 146096LL) / 146097LL;
    int64_t doe = z - era * 146097LL;
    int64_t yoe = (doe - doe / 1460LL + doe / 36524LL - doe / 146096LL) / 365LL;
    int64_t y = yoe + era * 400LL;
    int64_t doy = doe - (365LL * yoe + yoe / 4LL - yoe / 100LL);
    int64_t mp = (5LL * doy + 2LL) / 153LL;
    int64_t m = (mp < 10LL) ? (mp + 3LL) : (mp - 9LL);
    if (m <= 2LL) {
        y += 1LL;
    }
    return static_cast<int>(y);
}

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_code,
    const int32_t* __restrict__ cust_code,
    const int32_t* __restrict__ shipdate_days,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    int32_t base_year,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_sums[NUM_GROUPS];

    int tid = threadIdx.x;
    if (tid < NUM_GROUPS) {
        block_sums[tid] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n) {
        int32_t sd = shipdate_days[idx];
        if (sd >= date_lo && sd <= date_hi) {
            int32_t s = supp_code[idx];
            int32_t c = cust_code[idx];

            // SQL keeps only the two directed cross-nation pairs.
            if (((s == 0) && (c == 1)) || ((s == 1) && (c == 0))) {
                int yr = civil_year_from_days(sd);
                int yoff = yr - base_year;

                // Match the reference's fixed 8-key layout:
                // key = supp*4 + cust*2 + (year - 1995), years 1995/1996.
                if (yoff >= 0 && yoff < 2) {
                    int key = s * 4 + c * 2 + yoff;
                    double rev = price[idx] * (1.0 - discount[idx]);
                    atomicAdd(&block_sums[key], rev);
                }
            }
        }
    }

    __syncthreads();

    if (tid < NUM_GROUPS) {
        double v = block_sums[tid];
        if (v != 0.0) {
            atomicAdd(&out[tid], v);
        }
    }
}
