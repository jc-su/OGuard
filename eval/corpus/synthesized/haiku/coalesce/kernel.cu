#include <cstdint>

extern "C" __global__ void kernel(double* a, uint8_t* valid, double* b, double* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Check if row idx is valid: bit (idx % 8) of byte (idx / 8)
        int64_t byte_idx = idx / 8;
        int bit_idx = idx % 8;
        uint8_t is_valid = (valid[byte_idx] >> bit_idx) & 1;
        out[idx] = is_valid ? a[idx] : b[idx];
    }
}
