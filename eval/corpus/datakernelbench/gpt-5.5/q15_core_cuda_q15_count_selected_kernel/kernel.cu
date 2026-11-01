#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    const double* __restrict__ max_value,
    unsigned long long* __restrict__ count,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && present[i] != 0 && totals[i] == max_value[0]) {
        atomicAdd(count, 1ULL);
    }
}
