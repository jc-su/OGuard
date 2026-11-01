#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,          // not used, but kept for signature consistency
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    const int32_t* __restrict__ o_orderdate,
    const int32_t* __restrict__ o_shippriority,     // not used inside kernel
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    unsigned char* __restrict__ mask_out,           // torch bool --> uint8_t
    double* __restrict__ rev_out,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool keep = (l_shipdate[idx] > shipdate_lo) && (o_orderdate[idx] < orderdate_hi);
    mask_out[idx] = static_cast<unsigned char>(keep);
    rev_out[idx] = keep ? l_extendedprice[idx] * (1.0 - l_discount[idx]) : 0.0;
}
