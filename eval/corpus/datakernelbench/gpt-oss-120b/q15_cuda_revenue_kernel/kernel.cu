#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    const int32_t date_lo,
    const int32_t date_hi,
    const int32_t* __restrict__ supp_key_to_idx,
    double* __restrict__ supplier_totals,
    const int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t ship = l_shipdate[idx];
    if (ship < date_lo || ship >= date_hi) return;   // date filter

    int64_t supp = l_suppkey[idx];
    // map supplier key to compact index; -1 means not present in supplier table
    int32_t sup_idx = supp_key_to_idx[supp];
    if (sup_idx < 0) return;                         // no matching supplier

    double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
    atomicAdd(&supplier_totals[sup_idx], revenue);
}
