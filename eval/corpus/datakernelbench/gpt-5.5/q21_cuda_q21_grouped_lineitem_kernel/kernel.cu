#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const int64_t* __restrict__ l_suppkey,
    const int64_t* __restrict__ l_receipt,
    const int64_t* __restrict__ l_commit,
    int64_t n_line,
    const uint8_t* __restrict__ order_flags,
    int64_t max_orderkey,
    const int32_t* __restrict__ supp_map,
    int64_t max_suppkey,
    int64_t num_suppliers,
    int64_t* __restrict__ counts
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_line) {
        return;
    }

    int64_t order = l_orderkey[i];

    if (i > 0 && l_orderkey[i - 1] == order) {
        return;
    }

    if (order < 0 || order > max_orderkey || order_flags[order] == 0) {
        return;
    }

    int64_t end = i + 1;
    while (end < n_line && l_orderkey[end] == order) {
        ++end;
    }

    int32_t distinct_supp = 0;
    int32_t late_distinct_supp = 0;

    for (int64_t j = i; j < end; ++j) {
        int64_t sj = l_suppkey[j];
        bool seen_supplier = false;
        bool seen_late_supplier = false;

        for (int64_t k = i; k < j; ++k) {
            if (l_suppkey[k] == sj) {
                seen_supplier = true;
                if (l_receipt[k] > l_commit[k]) {
                    seen_late_supplier = true;
                }
            }
        }

        if (!seen_supplier) {
            ++distinct_supp;
        }
        if (l_receipt[j] > l_commit[j] && !seen_late_supplier) {
            ++late_distinct_supp;
        }
    }

    if (distinct_supp < 2 || late_distinct_supp > 1) {
        return;
    }

    for (int64_t j = i; j < end; ++j) {
        if (l_receipt[j] <= l_commit[j]) {
            continue;
        }

        int64_t sk = l_suppkey[j];
        if (sk < 0 || sk > max_suppkey) {
            continue;
        }

        int32_t sid = supp_map[sk];
        if (sid >= 0 && static_cast<int64_t>(sid) < num_suppliers) {
            atomicAdd(
                reinterpret_cast<unsigned long long int*>(&counts[sid]),
                1ULL
            );
        }
    }
}
