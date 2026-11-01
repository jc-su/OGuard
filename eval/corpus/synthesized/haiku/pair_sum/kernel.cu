#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* left_idx,
    const int32_t* right_idx,
    const double* left,
    const double* right,
    double* out,
    int64_t m,
    int64_t l,
    int64_t r
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        out[idx] = left[left_idx[idx]] + right[right_idx[idx]];
    }
}
