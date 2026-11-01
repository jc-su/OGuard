#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ order_keys,
    int64_t n,
    int64_t max_orderkey,
    uint8_t* __restrict__ flags
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int64_t k = order_keys[i];
    if (k >= 0 && k <= max_orderkey) {
        flags[k] = 1;
    }
}
