#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(double *run_values, int32_t *run_ends, double *out, int64_t r, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t left = 0, right = r - 1;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            // Explicit bounds check immediately before array access
            if (mid >= 0 && mid < r) {
                if (run_ends[mid] <= idx) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            } else {
                // Exit loop if mid is out of bounds
                break;
            }
        }
        
        // Final bounds check before accessing run_values
        if (left >= 0 && left < r) {
            out[idx] = run_values[left];
        }
    }
}
