#include <cstdint>

extern "C" __global__ void __launch_bounds__(256)
kernel(const int32_t* __restrict__ idx, const double* __restrict__ values, double* out,
       int64_t m, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < m) {
        atomicAdd(&out[idx[i]], values[i]);
    }
}
