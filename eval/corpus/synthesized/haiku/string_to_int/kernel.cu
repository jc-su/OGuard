#include <stdint.h>
#include <ctype.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, int64_t *out, uint8_t *valid, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        
        out[i] = 0;
        
        int32_t pos = start;
        int is_negative = 0;
        
        if (pos < end && (chars[pos] == '+' || chars[pos] == '-')) {
            is_negative = (chars[pos] == '-');
            pos++;
        }
        
        if (pos >= end) {
            return;
        }
        
        int64_t result = 0;
        for (int32_t j = pos; j < end; j++) {
            uint8_t c = chars[j];
            if (c < '0' || c > '9') {
                return;
            }
            result = result * 10 + (c - '0');
        }
        
        out[i] = is_negative ? -result : result;
        
        int64_t byte_idx = i / 8;
        int bit_idx = i % 8;
        valid[byte_idx] |= (uint8_t)(1 << bit_idx);
    }
}
