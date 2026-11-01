#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp,
    const int32_t* __restrict__ cust,
    const int32_t* __restrict__ shipday,
    const double* __restrict__ price,
    const double* __restrict__ disc,
    int32_t lo,
    int32_t hi,
    int32_t day1996_start,
    double* __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int32_t sd = shipday[idx];
    if (sd < lo || sd > hi) return;           // date filter

    int32_t s = supp[idx];
    int32_t c = cust[idx];
    // we still compute for all combos; the host will drop invalid ones later

    // year offset: 0 for 1995, 1 for 1996
    int32_t year_off = (sd >= day1996_start) ? 1 : 0;

    // key = (s * 2 + c) in [0,3]; out_idx = key*2 + year_off in [0,7]
    int32_t key = s * 2 + c;
    int32_t out_idx = key * 2 + year_off;

    double vol = price[idx] * (1.0 - disc[idx]);
    atomicAdd(&out[out_idx], vol);
}
