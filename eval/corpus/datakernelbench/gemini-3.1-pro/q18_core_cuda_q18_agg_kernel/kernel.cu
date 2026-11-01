#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    uint8_t* __restrict__ exists,
    int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t k = keys[i];
        double v = values[i];
        
        // Fast hardware-accelerated L2 atomic additions on Hopper architectures
        atomicAdd(&sums[k], v);
        exists[k] = 1;
    }
}
