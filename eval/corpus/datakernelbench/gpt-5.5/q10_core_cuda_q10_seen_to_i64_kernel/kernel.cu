#include <cstdint>

extern "C" __global__ void kernel(
    const uint8_t* __restrict__ seen,
    int64_t* __restrict__ prefix,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dense_size) {
        prefix[idx] = seen[idx] ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    }
}
