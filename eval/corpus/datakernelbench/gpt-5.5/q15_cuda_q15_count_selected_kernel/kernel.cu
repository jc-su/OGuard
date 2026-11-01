#include <stdint.h>

extern "C" __global__ void kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    int64_t* __restrict__ count_out,
    int64_t n_sup
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    double mx = max_value[0];
    if (present[idx] && totals[idx] == mx) {
        atomicAdd(reinterpret_cast<unsigned long long*>(count_out), 1ULL);
    }
}
