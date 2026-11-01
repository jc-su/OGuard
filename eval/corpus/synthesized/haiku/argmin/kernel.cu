#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *row, int64_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && n > 0) {
        int64_t min_val = values[0];
        int32_t min_row = 0;
        
        for (int64_t j = 1; j < n; j++) {
            if (values[j] < min_val) {
                min_val = values[j];
                min_row = j;
            }
        }
        
        row[0] = min_row;
    }
}
