#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *first, int64_t *out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t segment_start = first[i];
        int64_t max_val = values[segment_start];
        
        for (int64_t j = segment_start; j <= i; j++) {
            if (values[j] > max_val) {
                max_val = values[j];
            }
        }
        
        out[i] = max_val;
    }
}
