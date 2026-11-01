#include <stdint.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, uint32_t *out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        
        uint32_t h = 2166136261U;
        for (int32_t j = start; j < end; j++) {
            uint8_t c = chars[j];
            h = (h ^ c) * 16777619U;
        }
        
        out[i] = h;
    }
}
