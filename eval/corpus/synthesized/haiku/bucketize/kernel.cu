#include <cstdint>

extern "C" __global__ void kernel(double* values, double* bounds, int32_t* out, int64_t n, int64_t b) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double value = values[idx];
        // Linear search (since bounds are sorted)
        int64_t count = 0;
        for (int64_t j = 0; j < b; j++) {
            if (bounds[j] <= value) {
                count++;
            } else {
                break;
            }
        }
        out[idx] = (int32_t)count;
    }
}
