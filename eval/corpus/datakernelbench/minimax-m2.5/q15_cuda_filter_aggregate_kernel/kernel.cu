#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    const int64_t* __restrict__ sorted_suppkeys,
    double* __restrict__ out_revenue,
    int64_t n_lineitem,
    int n_supplier
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitem) return;

    int32_t shipdate = l_shipdate[idx];
    if (shipdate >= date_lo && shipdate < date_hi) {
        int64_t suppkey = l_suppkey[idx];
        // binary search in sorted supplier keys
        int64_t lo = 0, hi = n_supplier;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (sorted_suppkeys[mid] < suppkey) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < n_supplier && sorted_suppkeys[lo] == suppkey) {
            double revenue = l_price[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&out_revenue[lo], revenue);
        }
    }
}
