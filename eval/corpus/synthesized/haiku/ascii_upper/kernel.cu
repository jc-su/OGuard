#include <cstdint>

extern "C" __global__ void kernel(int32_t* offsets, uint8_t* chars, uint8_t* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_chars = offsets[n];

    if (idx < total_chars) {
        uint8_t c = chars[idx];
        // Convert lowercase a-z to uppercase A-Z
        if (c >= 97 && c <= 122) {  // 97 = 'a', 122 = 'z'
            out[idx] = c - 32;
        } else {
            out[idx] = c;
        }
    }
}
