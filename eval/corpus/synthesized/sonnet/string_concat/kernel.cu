#include <stdint.h>

#define WARPS_PER_BLOCK 8

// One warp per row; the lanes copy a's bytes, then b's bytes, with a stride of 32.
extern "C" __global__ void kernel(const int32_t *a_offsets, const uint8_t *a_chars,
                                  const int32_t *b_offsets, const uint8_t *b_chars,
                                  const int32_t *out_offsets, uint8_t *out_chars, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (i < n) {
        int64_t a_start = a_offsets[i];
        int64_t a_len = (int64_t)a_offsets[i + 1] - a_start;
        int64_t b_start = b_offsets[i];
        int64_t b_len = (int64_t)b_offsets[i + 1] - b_start;
        int64_t dst = out_offsets[i];
        for (int64_t k = lane; k < a_len; k += 32) {
            out_chars[dst + k] = a_chars[a_start + k];
        }
        for (int64_t k = lane; k < b_len; k += 32) {
            out_chars[dst + a_len + k] = b_chars[b_start + k];
        }
    }
}
