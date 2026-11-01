// Pair sum: out[i] = left[left_idx[i]] + right[right_idx[i]].
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* left_idx, const int32_t* right_idx,
                                  const double* left, const double* right, double* out,
                                  int64_t m, int64_t l, int64_t r) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
         i += stride) {
        out[i] = left[left_idx[i]] + right[right_idx[i]];
    }
}
