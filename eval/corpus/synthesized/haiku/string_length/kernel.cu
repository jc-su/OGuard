#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* offsets, const uint8_t* chars,
    int32_t* lengths)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = gridDim.x * blockDim.x - 1;
    if (i < n) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        int32_t count = 0;
        int32_t pos = start;
        while (pos < end) {
            uint8_t byte = chars[pos];
            if (byte < 128) {
                count++;
                pos++;
            } else if (byte < 224) {
                pos += 2;
                count++;
            } else if (byte < 240) {
                pos += 3;
                count++;
            } else {
                pos += 4;
                count++;
            }
        }
        lengths[i] = count;
    }
}
