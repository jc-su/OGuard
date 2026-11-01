#include <stdint.h>

extern "C" __global__ void kernel(uint32_t *packed, uint32_t *out, int64_t n, int64_t bits) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t bit_pos = i * bits;
        
        uint32_t code = 0;
        for (int64_t b = 0; b < bits; b++) {
            int64_t pos = bit_pos + b;
            int64_t w_idx = pos / 32;
            int bit_idx = pos % 32;
            uint32_t bit = (packed[w_idx] >> bit_idx) & 1;
            code |= (bit << b);
        }
        
        out[i] = code;
    }
}
