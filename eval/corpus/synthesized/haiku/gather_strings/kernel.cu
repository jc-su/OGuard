#include <stdint.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, int32_t *idx,
                                   int32_t *out_offsets, uint8_t *out_chars,
                                   int64_t n, int64_t m) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int32_t src_idx = idx[i];
        int32_t src_start = offsets[src_idx];
        int32_t src_end = offsets[src_idx + 1];
        int32_t dst_start = out_offsets[i];

        int32_t length = src_end - src_start;
        for (int32_t j = 0; j < length; j++) {
            out_chars[dst_start + j] = chars[src_start + j];
        }
    }
}
