#include <stdint.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ inv,
    const double* __restrict__ volume,
    const uint8_t* __restrict__ is_brazil,
    double* __restrict__ sum_total,
    double* __restrict__ sum_brazil,
    int64_t n,
    int32_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t g = inv[idx];
        double v = volume[idx];
        atomicAdd(&sum_total[g], v);
        if (is_brazil[idx]) {
            atomicAdd(&sum_brazil[g], v);
        }
    }
}
