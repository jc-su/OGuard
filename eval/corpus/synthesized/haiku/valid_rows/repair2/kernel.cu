#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void kernel(uint8_t *valid, int32_t *rows, int32_t *count, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t byte_idx = idx / 8;
        int bit_idx = idx % 8;
        
        // Load the byte with bounds checking
        if (byte_idx >= 0 && byte_idx < (n + 7) / 8) {
            uint8_t valid_byte = valid[byte_idx];
            
            // Check if bit is set (LSB first)
            uint8_t is_valid = (valid_byte >> bit_idx) & 1;
            
            if (is_valid) {
                // Atomically increment count and get position
                int32_t pos = atomicAdd(count, 1);
                // Explicit bounds check on position before storing
                if (pos >= 0 && pos < n) {
                    rows[pos] = (int32_t)idx;
                }
            }
        }
    }
}
