#include <stdint.h>

extern "C" __global__ void kernel(const int64_t* values, int64_t* out,
                                  int8_t* overflow, int64_t n, int64_t shift)
{
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    int k = shift >= 0 ? (int)shift : (int)(-shift);
    int64_t pw = 1;
    for (int j = 0; j < k; j++) pw *= 10;
    int64_t v = values[i];
    if (shift >= 0) {
        int64_t lim = INT64_MAX / pw;
        if (pw != 1 && (v > lim || v < -lim)) {
            out[i] = 0;
            overflow[i] = 1;
        } else {
            out[i] = (int64_t)((uint64_t)v * (uint64_t)pw);
            overflow[i] = 0;
        }
    } else {
        // C division truncates; turn it into floor division
        int64_t q = v / pw;
        int64_t r = v % pw;
        if (r < 0) {
            r += pw;
            q -= 1;
        }
        bool up = (v >= 0) ? (2 * r >= pw) : (2 * r > pw);
        out[i] = up ? q + 1 : q;
        overflow[i] = 0;
    }
}
