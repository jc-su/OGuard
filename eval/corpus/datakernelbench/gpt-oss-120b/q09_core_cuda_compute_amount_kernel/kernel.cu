#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ extprice,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double ep = extprice[idx];
        double d  = discount[idx];
        double sc = supplycost[idx];
        double q  = quantity[idx];
        out[idx] = ep * (1.0 - d) - sc * q;
    }
}
