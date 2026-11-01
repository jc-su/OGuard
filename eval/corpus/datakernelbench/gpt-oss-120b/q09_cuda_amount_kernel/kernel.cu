#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ out,
    const int64_t n)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    out[idx] = price[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
}
