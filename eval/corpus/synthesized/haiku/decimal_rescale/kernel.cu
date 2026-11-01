#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, int64_t *out, int8_t *overflow, int64_t n, int64_t shift) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t v = values[i];
        
        if (shift >= 0) {
            int64_t power = 1;
            for (int j = 0; j < shift; j++) power *= 10;
            
            __int128 result = (__int128)v * power;
            int64_t min_int64 = INT64_MIN;
            int64_t max_int64 = INT64_MAX;
            
            if (result < min_int64 || result > max_int64) {
                overflow[i] = 1;
                out[i] = 0;
            } else {
                overflow[i] = 0;
                out[i] = (int64_t)result;
            }
        } else {
            int64_t shift_abs = -shift;
            int64_t power = 1;
            for (int j = 0; j < shift_abs; j++) power *= 10;
            
            int64_t abs_v = (v < 0) ? -v : v;
            int64_t rounded = (abs_v + power / 2) / power;
            out[i] = (v < 0) ? -rounded : rounded;
            overflow[i] = 0;
        }
    }
}
