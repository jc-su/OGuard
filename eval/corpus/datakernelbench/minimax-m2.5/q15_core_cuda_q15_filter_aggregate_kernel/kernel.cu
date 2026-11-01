#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ sorted_suppkeys,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_totals,
    int64_t n_rows,
    int64_t n_suppliers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int32_t shipdate = l_shipdate_days[idx];
    if (shipdate < date_lo || shipdate >= date_hi) return;

    double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
    int64_t supp = l_suppkey[idx];

    // binary search lower_bound in sorted_suppkeys
    int64_t lo = 0, hi = n_suppliers;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (sorted_suppkeys[mid] < supp) lo = mid + 1;
        else hi = mid;
    }
    int64_t pos = lo;
    if (pos < n_suppliers && sorted_suppkeys[pos] == supp) {
        atomicAdd(&out_totals[pos], revenue);
    }
}
