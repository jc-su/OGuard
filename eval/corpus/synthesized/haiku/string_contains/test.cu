#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <string.h>
#include <assert.h>
#include "kernel.cu"

void cpu_reference(int32_t *offsets, uint8_t *chars, uint8_t *pattern, int8_t *match_ref, int64_t n, int64_t k) {
    if (k == 0) {
        for (int64_t i = 0; i < n; i++) {
            match_ref[i] = 1;
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            int32_t start = offsets[i];
            int32_t end = offsets[i + 1];
            int32_t str_len = end - start;
            if (str_len < k) {
                match_ref[i] = 0;
                continue;
            }
            bool found = false;
            for (int32_t j = 0; j <= str_len - k; j++) {
                bool all_match = true;
                for (int32_t l = 0; l < k; l++) {
                    if (chars[start + j + l] != pattern[l]) {
                        all_match = false;
                        break;
                    }
                }
                if (all_match) {
                    found = true;
                    break;
                }
            }
            match_ref[i] = found ? 1 : 0;
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int32_t *h_offsets = (int32_t*)malloc((n + 1) * sizeof(int32_t));
        h_offsets[0] = 0;
        int64_t total_bytes = 0;
        if (n > 0) {
            for (int64_t i = 0; i < n; i++) {
                int32_t str_len = 1 + (rand() % 20);
                h_offsets[i + 1] = h_offsets[i] + str_len;
            }
            total_bytes = h_offsets[n];
        }
        uint8_t *h_chars = (uint8_t*)malloc(total_bytes);
        int64_t k = n > 0 ? (rand() % 5) : 0;
        uint8_t *h_pattern = (uint8_t*)malloc(k);
        for (int64_t i = 0; i < total_bytes; i++) {
            h_chars[i] = (uint8_t)(rand() % 256);
        }
        for (int64_t i = 0; i < k; i++) {
            h_pattern[i] = (uint8_t)(rand() % 256);
        }
        int8_t *h_match = (int8_t*)malloc(n * sizeof(int8_t));
        int8_t *h_match_ref = (int8_t*)malloc(n * sizeof(int8_t));
        int32_t *d_offsets;
        uint8_t *d_chars;
        uint8_t *d_pattern;
        int8_t *d_match;
        cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_chars, total_bytes);
        cudaMalloc(&d_pattern, k);
        cudaMalloc(&d_match, n * sizeof(int8_t));
        cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chars, h_chars, total_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pattern, h_pattern, k, cudaMemcpyHostToDevice);
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_offsets, d_chars, d_pattern, d_match, n, k);
        cudaMemcpy(h_match, d_match, n * sizeof(int8_t), cudaMemcpyDeviceToHost);
        cpu_reference(h_offsets, h_chars, h_pattern, h_match_ref, n, k);
        for (int64_t i = 0; i < n; i++) {
            assert(h_match[i] == h_match_ref[i]);
        }
        printf("✓ n=%ld, k=%ld: passed\n", n, k);
        cudaFree(d_offsets);
        cudaFree(d_chars);
        cudaFree(d_pattern);
        cudaFree(d_match);
        free(h_offsets);
        free(h_chars);
        free(h_pattern);
        free(h_match);
        free(h_match_ref);
    }
    printf("All tests passed!\n");
    return 0;
}
