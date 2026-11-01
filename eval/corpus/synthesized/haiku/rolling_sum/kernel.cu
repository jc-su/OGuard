#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(double *values, double *out, int64_t n, int64_t window) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t start = idx >= window ? idx - window + 1 : 0;
        double total = 0.0;
        for (int64_t i = start; i <= idx; i++) {
            total += values[i];
        }
        out[idx] = total;
    }
}
