#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* offsets, const uint8_t* chars,
    const uint8_t* prefix, int64_t k,
    int8_t* match)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = gridDim.x * blockDim.x - 1;
    if (i < n) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        int32_t length = end - start;

        if (length < k) {
            match[i] = 0;
        } else {
            bool matches = true;
            for (int64_t j = 0; j < k; j++) {
                if (chars[start + j] != prefix[j]) {
                    matches = false;
                    break;
                }
            }
            match[i] = matches ? 1 : 0;
        }
    }
}
