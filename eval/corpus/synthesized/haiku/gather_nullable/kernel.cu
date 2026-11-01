#include <cstdint>
#include <cmath>

extern "C" __global__ void kernel(double* col, int32_t* idx, double* out, int64_t m, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int32_t index = idx[i];
        if (index == -1) {
            out[i] = NAN;
        } else {
            out[i] = col[index];
        }
    }
}
