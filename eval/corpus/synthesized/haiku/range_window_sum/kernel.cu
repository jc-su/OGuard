#include <stdint.h>

extern "C" __global__ void kernel(int64_t *ts, int64_t *values, int64_t *out, int64_t n, int64_t range_val) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t ts_i = ts[i];
        int64_t lower_bound = ts_i - range_val;
        
        int64_t total = 0;
        for (int64_t j = 0; j < n; j++) {
            if (ts[j] >= lower_bound && ts[j] <= ts_i) {
                total += values[j];
            }
        }
        
        out[i] = total;
    }
}
