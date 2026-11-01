#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(double *run_values, int32_t *run_ends, double *out, int64_t r, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t left = 0, right = r - 1;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            if (run_ends[mid] <= idx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        // Clamp left to valid range
        left = (left >= r) ? r - 1 : left;
        if (left >= 0 && left < r) {
            out[idx] = run_values[left];
        }
    }
}
