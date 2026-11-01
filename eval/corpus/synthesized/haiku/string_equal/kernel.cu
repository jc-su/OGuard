#include <cstdint>

extern "C" __global__ void kernel(int32_t* a_offsets, uint8_t* a_chars, int32_t* b_offsets, uint8_t* b_chars, int8_t* match, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Get string offsets and lengths
        int32_t a_start = a_offsets[i];
        int32_t a_end = a_offsets[i + 1];
        int32_t a_len = a_end - a_start;

        int32_t b_start = b_offsets[i];
        int32_t b_end = b_offsets[i + 1];
        int32_t b_len = b_end - b_start;

        // Check if lengths differ
        if (a_len != b_len) {
            match[i] = 0;
        } else {
            // Compare bytes
            bool is_equal = true;
            for (int32_t j = 0; j < a_len; j++) {
                if (a_chars[a_start + j] != b_chars[b_start + j]) {
                    is_equal = false;
                    break;
                }
            }
            match[i] = is_equal ? 1 : 0;
        }
    }
}
