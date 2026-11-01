#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, uint8_t *valid, uint8_t *out,
                                   int64_t n, int64_t lo, int64_t hi) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t byte_idx = i / 8;
        int bit_idx = i % 8;
        uint8_t valid_bit = (valid[byte_idx] >> bit_idx) & 1;

        uint8_t mask = 1 << bit_idx;
        if (valid_bit && values[i] >= lo && values[i] <= hi) {
            out[byte_idx] |= mask;
        } else {
            out[byte_idx] &= ~mask;
        }
    }
}
