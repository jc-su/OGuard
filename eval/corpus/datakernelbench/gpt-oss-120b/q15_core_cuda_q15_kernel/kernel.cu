#include <cuda_runtime.h>

__device__ __forceinline__ int64_t lower_bound(const int64_t* data,
                                               int64_t size,
                                               int64_t key) {
    int64_t lo = 0;
    int64_t hi = size;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = data[mid];
        if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

extern "C" __global__ void kernel(const int64_t* __restrict__ suppkey,
                           const double* __restrict__ price,
                           const double* __restrict__ disc,
                           const int32_t* __restrict__ shipdate,
                           const int64_t* __restrict__ sorted_supp,
                           double* __restrict__ totals,
                           int64_t* __restrict__ counts,
                           int32_t date_lo,
                           int32_t date_hi,
                           int64_t n_rows,
                           int64_t n_sup) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    // Apply shipdate filter.
    int32_t sd = shipdate[idx];
    if (sd < date_lo || sd >= date_hi) return;

    // Compute revenue for this lineitem row.
    double rev = price[idx] * (1.0 - disc[idx]);

    // Map l_suppkey to supplier index via binary search.
    int64_t key = suppkey[idx];
    int64_t pos = lower_bound(sorted_supp, n_sup, key);
    if (pos < n_sup && sorted_supp[pos] == key) {
        // Atomically accumulate revenue and row count.
        atomicAdd(&totals[pos], rev);
        // atomicAdd for signed 64‑bit uses unsigned long long.
        atomicAdd((unsigned long long*)&counts[pos], 1ULL);
    }
}
