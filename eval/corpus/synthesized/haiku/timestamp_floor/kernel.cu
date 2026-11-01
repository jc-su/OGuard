#include <stdint.h>

extern "C" __global__ void kernel(int64_t *ts, int64_t *out, int64_t n, int64_t unit) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t ts_val = ts[i];
        if (ts_val >= 0) {
            out[i] = (ts_val / unit) * unit;
        } else {
            if (ts_val % unit == 0) {
                out[i] = ts_val;
            } else {
                out[i] = ((ts_val / unit) - 1) * unit;
            }
        }
    }
}
