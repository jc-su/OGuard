#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] * b[idx];
    }
}
