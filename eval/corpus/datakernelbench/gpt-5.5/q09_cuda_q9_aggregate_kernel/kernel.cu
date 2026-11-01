#include <cstdint>

#define MAX_GROUPS 2048

extern "C" __global__ void kernel(
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    const int64_t* __restrict__ nationkey,
    const int32_t* __restrict__ year,
    int64_t n_elements,
    int32_t year_min,
    int32_t n_years,
    int64_t max_nation_key,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ out_counts
) {
    __shared__ double sh_sums[MAX_GROUPS];
    __shared__ unsigned long long sh_counts[MAX_GROUPS];

    const int tid = threadIdx.x;
    const int group_count = static_cast<int>((max_nation_key + 1) * static_cast<int64_t>(n_years));

    for (int g = tid; g < group_count; g += blockDim.x) {
        sh_sums[g] = 0.0;
        sh_counts[g] = 0ULL;
    }
    __syncthreads();

    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += stride) {
        const int64_t nk = nationkey[idx];
        const int32_t yr = year[idx];
        const int32_t yoff = yr - year_min;

        if (nk >= 0 && nk <= max_nation_key && yoff >= 0 && yoff < n_years) {
            const int64_t group64 = nk * static_cast<int64_t>(n_years) + static_cast<int64_t>(yoff);
            if (group64 >= 0 && group64 < group_count) {
                const int group = static_cast<int>(group64);
                const double amount =
                    extendedprice[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
                atomicAdd(&sh_sums[group], amount);
                atomicAdd(&sh_counts[group], 1ULL);
            }
        }
    }

    __syncthreads();

    for (int g = tid; g < group_count; g += blockDim.x) {
        const unsigned long long c = sh_counts[g];
        if (c != 0ULL) {
            atomicAdd(&out_sums[g], sh_sums[g]);
            atomicAdd(&out_counts[g], c);
        }
    }
}
