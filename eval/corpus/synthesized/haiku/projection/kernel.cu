#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const double* price, const double* discount, int64_t n,
    double* out)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = price[i] * (1.0 - discount[i]);
    }
}
