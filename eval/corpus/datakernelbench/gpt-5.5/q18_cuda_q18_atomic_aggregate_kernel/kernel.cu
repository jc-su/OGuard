#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double* __restrict__ sums,
    int64_t n,
    int64_t sums_n
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        int64_t k = keys[idx];
        if (k >= 0 && k < sums_n) {
            atomicAdd(&sums[k], qty[idx]);
        }
    }
}
