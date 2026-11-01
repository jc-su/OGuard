#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ values,
    int64_t target,
    bool* __restrict__ out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (values[idx] == target);
    }
}
