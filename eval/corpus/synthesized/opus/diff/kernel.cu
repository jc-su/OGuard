#include <cstdint>

extern "C" __global__ void __launch_bounds__(256)
kernel(const int64_t* __restrict__ values, int64_t* __restrict__ out, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i == 0 ? values[0] : values[i] - values[i - 1];
    }
}
