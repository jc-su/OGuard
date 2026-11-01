#include <stdint.h>

extern "C" __global__ void kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    int64_t* __restrict__ counter,
    int64_t* __restrict__ selected_indices,
    double* __restrict__ selected_revenues,
    int64_t n_sup
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    double mx = max_value[0];
    if (present[idx] && totals[idx] == mx) {
        unsigned long long pos = atomicAdd(
            reinterpret_cast<unsigned long long*>(counter),
            1ULL
        );
        selected_indices[static_cast<int64_t>(pos)] = idx;
        selected_revenues[static_cast<int64_t>(pos)] = totals[idx];
    }
}
