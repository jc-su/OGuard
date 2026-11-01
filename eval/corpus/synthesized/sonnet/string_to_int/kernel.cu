#include <stdint.h>

// One thread per string. The validity bits of the 32 rows of a warp are
// combined with a ballot, and lanes 0 to 3 each write one byte of the bitmap.
extern "C" __global__ void kernel(const int32_t *offsets, const uint8_t *chars, int64_t *out,
                                  uint8_t *valid, int64_t n) {
    int lane = threadIdx.x & 31;
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    bool ok = false;
    int64_t value = 0;
    if (i < n) {
        int64_t p = offsets[i];
        int64_t end = offsets[i + 1];
        bool negative = false;
        if (p < end) {
            uint8_t first = chars[p];
            if (first == '+') {
                p++;
            } else if (first == '-') {
                negative = true;
                p++;
            }
        }
        // The magnitude of INT64_MIN is one more than INT64_MAX.
        uint64_t limit = 9223372036854775807ull + (negative ? 1 : 0);
        uint64_t acc = 0;
        ok = p < end;
        while (ok && p < end) {
            int digit = (int)chars[p] - '0';
            if (digit < 0 || digit > 9) {
                ok = false;
            } else if (acc > (limit - digit) / 10) {
                ok = false;
            } else {
                acc = acc * 10 + digit;
                p++;
            }
        }
        if (ok) {
            value = (int64_t)(negative ? 0ull - acc : acc);
        }
        out[i] = value;
    }
    unsigned bits = __ballot_sync(0xFFFFFFFFu, ok);
    int64_t byte = (((int64_t)blockIdx.x * 256 + threadIdx.x - lane) >> 3) + lane;
    if (lane < 4 && byte < (n + 7) / 8) {
        valid[byte] = (uint8_t)((bits >> (8 * lane)) & 0xFF);
    }
}
