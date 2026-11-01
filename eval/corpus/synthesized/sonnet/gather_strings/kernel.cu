#include <stdint.h>

#define WARPS_PER_BLOCK 8

// One warp per output string; the lanes copy its bytes with a stride of 32.
extern "C" __global__ void kernel(const int32_t *offsets, const uint8_t *chars,
                                  const int32_t *idx, const int32_t *out_offsets,
                                  uint8_t *out_chars, int64_t n, int64_t m) {
    int64_t i = (int64_t)blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (i < m) {
        int64_t src = idx[i];
        int64_t start = offsets[src];
        int64_t length = (int64_t)offsets[src + 1] - start;
        int64_t dst = out_offsets[i];
        for (int64_t k = lane; k < length; k += 32) {
            out_chars[dst + k] = chars[start + k];
        }
    }
}
