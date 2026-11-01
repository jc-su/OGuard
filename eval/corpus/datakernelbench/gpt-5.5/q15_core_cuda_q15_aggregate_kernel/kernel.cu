#include <cstdint>

__device__ __forceinline__ int64_t q15_find_supplier_index(
    int64_t supp,
    const int64_t* __restrict__ s_suppkey,
    int64_t n_sup
) {
    int64_t direct = supp - 1;
    if (direct >= 0 && direct < n_sup && s_suppkey[direct] == supp) {
        return direct;
    }

    int64_t lo = 0;
    int64_t hi = n_sup;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        int64_t v = s_suppkey[mid];
        if (v < supp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n_sup && s_suppkey[lo] == supp) {
        return lo;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ s_suppkey,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ totals,
    uint8_t* __restrict__ present,
    int64_t n_line,
    int64_t n_sup
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_line) {
        return;
    }

    int32_t sd = l_shipdate_days[i];
    if (sd >= shipdate_lo_days && sd < shipdate_hi_days) {
        int64_t supp = l_suppkey[i];
        int64_t out_idx = q15_find_supplier_index(supp, s_suppkey, n_sup);
        if (out_idx >= 0) {
            double revenue = l_extendedprice[i] * (1.0 - l_discount[i]);
            atomicAdd(totals + out_idx, revenue);
            present[out_idx] = static_cast<uint8_t>(1);
        }
    }
}
