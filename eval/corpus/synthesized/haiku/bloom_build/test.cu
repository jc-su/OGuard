#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t* keys, uint32_t* bits, int64_t n, int64_t words);

uint64_t hash_bloom_h1(int64_t key) {
    return ((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32;
}

uint64_t hash_bloom_h2(int64_t key) {
    return ((uint64_t)key * 0xC2B2AE3D27D4EB4F) >> 32;
}

void test_bloom_build(int64_t n, int64_t words) {
    printf("Testing bloom_build with n=%ld, words=%ld\n", n, words);

    // Allocate host memory
    int64_t* h_keys = new int64_t[n];
    uint32_t* h_bits = new uint32_t[words];
    uint32_t* h_expected = new uint32_t[words];

    // Initialize
    for (int64_t i = 0; i < n; i++) {
        h_keys[i] = i;
    }

    for (int64_t i = 0; i < words; i++) {
        h_bits[i] = 0;
        h_expected[i] = 0;
    }

    // Compute expected result
    for (int64_t i = 0; i < n; i++) {
        uint64_t h1 = hash_bloom_h1(h_keys[i]);
        uint64_t h2 = hash_bloom_h2(h_keys[i]);

        int64_t bit_count = 32 * words;
        int64_t bit_pos1 = h1 % bit_count;
        int64_t bit_pos2 = h2 % bit_count;

        int64_t word_idx1 = bit_pos1 / 32;
        int bit_idx1 = bit_pos1 % 32;
        h_expected[word_idx1] |= (1u << bit_idx1);

        if (bit_pos2 != bit_pos1) {
            int64_t word_idx2 = bit_pos2 / 32;
            int bit_idx2 = bit_pos2 % 32;
            h_expected[word_idx2] |= (1u << bit_idx2);
        }
    }

    // Allocate device memory
    int64_t* d_keys = nullptr;
    uint32_t* d_bits = nullptr;
    cudaMalloc(&d_keys, n * sizeof(int64_t));
    cudaMalloc(&d_bits, words * sizeof(uint32_t));

    // Copy to device
    cudaMemcpy(d_keys, h_keys, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bits, h_bits, words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_keys, d_bits, n, words);

    // Copy result back
    cudaMemcpy(h_bits, d_bits, words * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < words; i++) {
        if (h_bits[i] != h_expected[i]) {
            pass = false;
            if (words <= 16) {
                printf("  Mismatch at word %ld: got %u, expected %u\n", i, h_bits[i], h_expected[i]);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_bits);
    delete[] h_keys;
    delete[] h_bits;
    delete[] h_expected;
}

int main() {
    int64_t test_configs[][2] = {{0, 4}, {1, 4}, {10, 8}, {1000, 16}};
    for (int i = 0; i < 4; i++) {
        test_bloom_build(test_configs[i][0], test_configs[i][1]);
    }
    return 0;
}
