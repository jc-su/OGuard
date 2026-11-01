#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* source,
    int32_t* out,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx < total) {
        int64_t i = idx / cols;
        int64_t j = idx % cols;
        out[j * rows + i] = source[i * cols + j];
    }
}
