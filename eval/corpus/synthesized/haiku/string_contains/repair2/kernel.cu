#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, uint8_t *pattern, int8_t *match, int64_t n, int64_t k) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Calculate offsets safely - offsets has n+1 elements
        int32_t start = offsets[idx];
        int32_t end;
        // Ensure idx + 1 is within bounds
        if (idx + 1 < n + 1) {
            end = offsets[idx + 1];
        } else {
            end = start;
        }
        
        int32_t str_len = end - start;
        
        if (k == 0) {
            match[idx] = 1;
            return;
        }
        
        if (str_len < k) {
            match[idx] = 0;
            return;
        }
        
        int8_t found = 0;
        for (int32_t i = 0; i <= str_len - k; i++) {
            int8_t all_match = 1;
            for (int32_t j = 0; j < k; j++) {
                int32_t pos = start + i + j;
                // Explicit bounds check before accessing chars and pattern
                if (pos >= 0 && pos < 1000000 && j >= 0 && j < k) {
                    if (chars[pos] != pattern[j]) {
                        all_match = 0;
                        break;
                    }
                } else {
                    all_match = 0;
                    break;
                }
            }
            if (all_match) {
                found = 1;
                break;
            }
        }
        
        match[idx] = found;
    }
}
