#include <stdint.h>

extern "C" __global__ void kernel(
    int64_t* __restrict__ ptr,
    int64_t n,
    int64_t value
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}
