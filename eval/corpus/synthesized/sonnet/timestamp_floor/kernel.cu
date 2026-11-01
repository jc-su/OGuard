#include <stdint.h>

extern "C" __global__ void kernel(const int64_t *ts, int64_t *out, int64_t n, int64_t unit) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        int64_t t = ts[i];
        int64_t r = t % unit;
        if (r < 0) {
            r += unit;
        }
        out[i] = t - r;
    }
}
