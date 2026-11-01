#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n,
    int64_t* __restrict__ out_suppkey,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t sd = l_shipdate_days[idx];
    if (sd >= date_lo && sd < date_hi) {
        int64_t sk = l_suppkey[idx];
        double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        int32_t pos = atomicAdd(out_count, 1);
        out_suppkey[pos] = sk;
        out_revenue[pos] = rev;
    }
}
