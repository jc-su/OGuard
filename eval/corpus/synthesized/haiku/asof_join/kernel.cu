#include <stdint.h>

extern "C" __global__ void kernel(int64_t *left, int64_t *right, int32_t *out,
                                   int64_t n, int64_t r, int64_t tolerance) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t left_val = left[i];
        int32_t best_idx = -1;

        for (int64_t j = 0; j < r; j++) {
            if (right[j] <= left_val && left_val - right[j] <= tolerance) {
                best_idx = j;
            }
        }

        out[i] = best_idx;
    }
}
