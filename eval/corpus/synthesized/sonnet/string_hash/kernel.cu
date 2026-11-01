#include <stdint.h>

extern "C" __global__ void kernel(const int32_t* offsets, const uint8_t* chars,
                                  uint32_t* out, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    uint32_t h = 2166136261u;
    int32_t end = offsets[i + 1];
    for (int32_t p = offsets[i]; p < end; p++) {
        h = (h ^ (uint32_t)chars[p]) * 16777619u;
    }
    out[i] = h;
}
