#include <stdint.h>

extern "C" __global__ void kernel(int32_t *a_offsets, uint8_t *a_chars, int32_t *b_offsets, uint8_t *b_chars,
                                   int32_t *out_offsets, uint8_t *out_chars, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t a_start = a_offsets[i];
        int32_t a_end = a_offsets[i + 1];
        int32_t b_start = b_offsets[i];
        int32_t b_end = b_offsets[i + 1];
        
        int32_t out_start = out_offsets[i];
        
        int32_t a_len = a_end - a_start;
        int32_t b_len = b_end - b_start;
        
        for (int32_t j = 0; j < a_len; j++) {
            out_chars[out_start + j] = a_chars[a_start + j];
        }
        
        for (int32_t j = 0; j < b_len; j++) {
            out_chars[out_start + a_len + j] = b_chars[b_start + j];
        }
    }
}
