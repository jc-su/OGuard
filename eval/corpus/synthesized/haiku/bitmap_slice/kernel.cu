#include <stdint.h>

extern "C" __global__ void kernel(uint8_t *source, uint8_t *out, int64_t n, int64_t offset) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t src_byte_idx = (offset + i) / 8;
        int src_bit_idx = (offset + i) % 8;
        uint8_t src_bit = (source[src_byte_idx] >> src_bit_idx) & 1;
        
        int64_t out_byte_idx = i / 8;
        int out_bit_idx = i % 8;
        
        uint8_t mask = 1 << out_bit_idx;
        if (src_bit) {
            out[out_byte_idx] |= mask;
        } else {
            out[out_byte_idx] &= ~mask;
        }
    }
}
