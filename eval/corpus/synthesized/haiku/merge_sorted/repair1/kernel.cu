#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t *a, int64_t *b, int64_t *out, int64_t m, int64_t p) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = m + p;

    if (idx < total) {
        int64_t i = 0, j = 0;
        
        for (int64_t k = 0; k < idx; k++) {
            if (i >= m) {
                j++;
            } else if (j >= p) {
                i++;
            } else if (i < m && j < p && a[i] <= b[j]) {
                i++;
            } else {
                j++;
            }
        }
        
        // Ensure valid bounds for array access
        if (i >= m && j < p) {
            out[idx] = b[j];
        } else if (j >= p && i < m) {
            out[idx] = a[i];
        } else if (i < m && j < p) {
            if (a[i] <= b[j]) {
                out[idx] = a[i];
            } else {
                out[idx] = b[j];
            }
        }
    }
}
