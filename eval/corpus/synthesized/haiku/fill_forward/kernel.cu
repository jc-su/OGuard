#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, uint8_t *valid, int64_t *out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t last_valid = 0;
        int found = 0;
        
        for (int64_t j = 0; j <= i; j++) {
            int64_t byte_idx = j / 8;
            int bit_idx = j % 8;
            int valid_bit = (valid[byte_idx] >> bit_idx) & 1;
            
            if (valid_bit) {
                last_valid = values[j];
                found = 1;
            }
        }
        
        out[i] = found ? last_valid : 0;
    }
}
