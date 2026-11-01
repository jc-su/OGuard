#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, uint8_t *pattern, int8_t *match, int64_t n, int64_t k) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t start = offsets[idx];
        int32_t end = offsets[idx + 1];
        int32_t str_len = end - start;
        
        if (k == 0) {
            match[idx] = 1;
            return;
        }
        
        if (str_len < k) {
            match[idx] = 0;
            return;
        }
        
        bool found = false;
        for (int32_t i = 0; i <= str_len - k; i++) {
            bool all_match = true;
            for (int32_t j = 0; j < k; j++) {
                if (chars[start + i + j] != pattern[j]) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                found = true;
                break;
            }
        }
        
        match[idx] = found ? 1 : 0;
    }
}
