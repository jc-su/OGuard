#include <cstdint>
#include <cmath>

extern "C" __global__ void kernel(double* values, int32_t* first, double* out, int64_t n, int64_t k) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t lag_idx = i - k;
        int32_t first_row = first[i];

        if (lag_idx >= first_row) {
            out[i] = values[lag_idx];
        } else {
            out[i] = NAN;
        }
    }
}
