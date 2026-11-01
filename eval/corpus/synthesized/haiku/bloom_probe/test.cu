#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t* keys, uint32_t* bits, int8_t* match, int64_t n, int64_t words);

uint64_t hash_bloom_h1(int64_t key) {
    return ((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32;
}

uint64_t hash_bloom_h2(int64_t key) {
    return ((uint64_t)key * 0xC2B2AE3D27D4EB4F) >> 32;
}

void test_bloom_probe(int64_t n, int64_t words) {
    printf("Testing bloom_probe with n=%ld, words=%ld\n", n, words);

    // Build filter with some keys
    int64_t build_count = n / 2;
    int64_t* h_build_keys = new int64_t[build_count];
    uint32_t* h_bits = new uint32_t[words];

    for (int64_t i = 0; i < build_count; i++) {
        h_build_keys[i] = i;
    }

    for (int64_t i = 0; i < words; i++) {
        h_bits[i] = 0;
    }

    // Compute bits
    for (int64_t i = 0; i < build_count; i++) {
        uint64_t h1 = hash_bloom_h1(h_build_keys[i]);
        uint64_t h2 = hash_bloom_h2(h_build_keys[i]);

        int64_t bit_count = 32 * words;
        int64_t bit_pos1 = h1 % bit_count;
        int64_t bit_pos2 = h2 % bit_count;

        int64_t word_idx1 = bit_pos1 / 32;
        int bit_idx1 = bit_pos1 % 32;
        h_bits[word_idx1] |= (1u << bit_idx1);

        if (bit_pos2 != bit_pos1) {
            int64_t word_idx2 = bit_pos2 / 32;
            int bit_idx2 = bit_pos2 % 32;
            h_bits[word_idx2] |= (1u << bit_idx2);
        }
    }

    // Probe with different keys
    int64_t* h_probe_keys = new int64_t[n];
    int8_t* h_match = new int8_t[n];

    for (int64_t i = 0; i < n; i++) {
        h_probe_keys[i] = i;
    }

    // Allocate device memory
    int64_t* d_probe_keys = nullptr;
    uint32_t* d_bits = nullptr;
    int8_t* d_match = nullptr;
    cudaMalloc(&d_probe_keys, n * sizeof(int64_t));
    cudaMalloc(&d_bits, words * sizeof(uint32_t));
    cudaMalloc(&d_match, n * sizeof(int8_t));

    // Copy to device
    cudaMemcpy(d_probe_keys, h_probe_keys, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bits, h_bits, words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_probe_keys, d_bits, d_match, n, words);

    // Copy result back
    cudaMemcpy(h_match, d_match, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Verify: all built keys should be found (no false negatives)
    bool pass = true;
    for (int64_t i = 0; i < build_count; i++) {
        if (h_match[i] != 1) {
            pass = false;
            if (n <= 10) {
                printf("  False negative for key %ld\n", h_probe_keys[i]);
            }
        }
    }

    if (pass) {
        printf("  PASS (no false negatives)\n");
    } else {
        printf("  FAIL (false negatives detected)\n");
    }

    // Cleanup
    cudaFree(d_probe_keys);
    cudaFree(d_bits);
    cudaFree(d_match);
    delete[] h_build_keys;
    delete[] h_bits;
    delete[] h_probe_keys;
    delete[] h_match;
}

int main() {
    int64_t test_configs[][2] = {{0, 4}, {1, 4}, {10, 8}, {1000, 16}};
    for (int i = 0; i < 4; i++) {
        test_bloom_probe(test_configs[i][0], test_configs[i][1]);
    }
    return 0;
}
