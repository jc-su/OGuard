#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    int64_t n,
    double* __restrict__ sums,
    int32_t* __restrict__ flags,
    int64_t key_range
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        if (key >= 0 && key < key_range) {
            atomicAdd(&sums[key], qty[idx]);
            atomicExch(&flags[key], 1);
        }
    }
}
