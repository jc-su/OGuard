#include <stdint.h>

// One thread per code. A code spans two words when its bits cross a word boundary.
extern "C" __global__ void kernel(const uint32_t *packed, uint32_t *out, int64_t n, int64_t w,
                                  int64_t bits) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        int64_t p = i * bits;
        int64_t word = p >> 5;
        int shift = (int)(p & 31);
        uint64_t v = packed[word];
        if (shift + bits > 32) {
            v |= (uint64_t)packed[word + 1] << 32;
        }
        v >>= shift;
        uint64_t mask = (1ull << bits) - 1;
        out[i] = (uint32_t)(v & mask);
    }
}
