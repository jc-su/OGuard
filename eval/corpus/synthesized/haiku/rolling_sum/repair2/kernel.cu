#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(double *values, double *out, int64_t n, int64_t window) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t start = idx >= window ? idx - window + 1 : 0;
        // Explicit bounds check on start
        if (start < 0) start = 0;
        if (start >= n) start = n - 1;
        
        double total = 0.0;
        // Loop only while i is within bounds
        for (int64_t i = start; i <= idx && i < n; i++) {
            // Bounds check immediately before access
            if (i >= 0 && i < n) {
                total += values[i];
            }
        }
        out[idx] = total;
    }
}
