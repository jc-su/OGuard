#include <cstdint>

extern "C" __global__ void kernel(const int64_t* values, int8_t* flags, int64_t n) {
    int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start; i < n; i += stride) {
        flags[i] = (i == 0 || values[i] != values[i - 1]) ? 1 : 0;
    }
}
