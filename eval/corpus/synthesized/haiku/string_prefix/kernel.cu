#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* offsets,
    const uint8_t* chars,
    uint8_t* out,
    int64_t n,
    int64_t k
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t start = offsets[idx];
        int32_t end = offsets[idx + 1];
        int32_t length = (end - start < k) ? (end - start) : k;

        // Copy string bytes
        for (int32_t j = 0; j < length; j++) {
            out[idx * k + j] = chars[start + j];
        }

        // Pad with zeros
        for (int32_t j = length; j < k; j++) {
            out[idx * k + j] = 0;
        }
    }
}
