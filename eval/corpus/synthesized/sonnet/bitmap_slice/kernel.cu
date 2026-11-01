#include <stdint.h>

extern "C" __global__ void kernel(const uint8_t* source, uint8_t* out, int64_t n, int64_t offset)
{
    int64_t j = (int64_t)blockIdx.x * 256 + threadIdx.x;
    int64_t nbytes = (n + 7) / 8;
    if (j >= nbytes) return;
    int64_t s = offset >> 3;
    int b = (int)(offset & 7);
    int64_t source_bytes = (offset + n + 7) / 8;
    unsigned value = (unsigned)source[s + j] >> b;
    if (b != 0 && s + j + 1 < source_bytes) {
        value |= (unsigned)source[s + j + 1] << (8 - b);
    }
    value &= 0xFFu;
    int64_t remaining = n - 8 * j;
    if (remaining < 8) {
        value &= (1u << (int)remaining) - 1u;
    }
    out[j] = (uint8_t)value;
}
