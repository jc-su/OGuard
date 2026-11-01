#include <cstdint>

// The difference is computed in unsigned arithmetic so that it wraps on overflow.
extern "C" __global__ void kernel(const int64_t* values, int64_t* out, int64_t n) {
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start; i < n; i += stride) {
        out[i] = i == 0 ? values[0] : (int64_t)((uint64_t)values[i] - (uint64_t)values[i - 1]);
    }
}
