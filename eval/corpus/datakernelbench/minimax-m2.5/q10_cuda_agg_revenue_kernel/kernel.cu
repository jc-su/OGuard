#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        double rev = price[idx] * (1.0 - discount[idx]);
        atomicAdd(&out[key], rev);
    }
}
