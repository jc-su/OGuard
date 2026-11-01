#include <cstdint>

extern "C" __global__ void kernel(const int32_t* idx, const double* values, double* out, int64_t m,
                                  int64_t n) {
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start; i < m; i += stride) atomicAdd(&out[idx[i]], values[i]);
}
