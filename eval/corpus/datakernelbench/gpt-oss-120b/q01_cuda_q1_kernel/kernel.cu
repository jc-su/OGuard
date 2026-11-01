#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ rf_code,
    const int32_t* __restrict__ ls_code,
    const double*  __restrict__ qty,
    const double*  __restrict__ price,
    const double*  __restrict__ disc,
    const double*  __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    const int32_t shipdate_hi,
    double* __restrict__ sum_qty,
    double* __restrict__ sum_base_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    unsigned long long* __restrict__ cnt,
    const int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Date predicate (<= shipdate_hi)
    int32_t sday = shipdate_days[idx];
    if (sday > shipdate_hi) return;

    // Compute group index
    int32_t grp = rf_code[idx] * 256 + ls_code[idx];
    // Load required columns
    double q   = qty[idx];
    double p   = price[idx];
    double d   = disc[idx];
    double t   = tax[idx];

    double disc_price = p * (1.0 - d);
    double charge     = disc_price * (1.0 + t);

    // Atomic accumulation (FP64 atomics are supported on sm_90)
    atomicAdd(&sum_qty[grp],          q);
    atomicAdd(&sum_base_price[grp],   p);
    atomicAdd(&sum_disc_price[grp],   disc_price);
    atomicAdd(&sum_charge[grp],       charge);
    atomicAdd(&sum_disc[grp],         d);
    atomicAdd(&cnt[grp], (unsigned long long)1);
}
