#include <cstdint>

extern "C" __global__ void kernel(int64_t* keys, uint32_t* bits, int8_t* match, int64_t n, int64_t words) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];

        // Hash functions
        uint64_t h1 = ((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32;
        uint64_t h2 = ((uint64_t)key * 0xC2B2AE3D27D4EB4F) >> 32;

        // Check bits
        int64_t bit_count = 32 * words;
        int64_t bit_pos1 = h1 % bit_count;
        int64_t bit_pos2 = h2 % bit_count;

        int64_t word_idx1 = bit_pos1 / 32;
        int bit_idx1 = bit_pos1 % 32;
        int is_set1 = (bits[word_idx1] >> bit_idx1) & 1;

        int64_t word_idx2 = bit_pos2 / 32;
        int bit_idx2 = bit_pos2 % 32;
        int is_set2 = (bits[word_idx2] >> bit_idx2) & 1;

        match[idx] = (is_set1 && is_set2) ? 1 : 0;
    }
}
