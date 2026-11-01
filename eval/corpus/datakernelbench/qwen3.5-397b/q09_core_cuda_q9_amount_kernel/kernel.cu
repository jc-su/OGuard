#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ amount,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double p = price[idx];
        double d = discount[idx];
        double s = supplycost[idx];
        double q = quantity[idx];
        amount[idx] = p * (1.0 - d) - s * q;
    }
}
