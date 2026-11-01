#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int32_t *group, int64_t *value, uint8_t *valid,
                                   int64_t *counts, int64_t *sums, int64_t n, int64_t g);

void cpu_ref(int32_t *group, int64_t *value, uint8_t *valid,
             int64_t *counts, int64_t *sums, int64_t n, int64_t g) {
    memset(counts, 0, g * sizeof(int64_t));
    memset(sums, 0, g * sizeof(int64_t));

    for (int64_t i = 0; i < n; i++) {
        int64_t byte_idx = i / 8;
        int bit_idx = i % 8;
        uint8_t valid_bit = (valid[byte_idx] >> bit_idx) & 1;

        if (valid_bit) {
            int32_t grp = group[i];
            counts[grp]++;
            sums[grp] += value[i];
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int num_tests = 4;
    int64_t g = 10;

    for (int t = 0; t < num_tests; t++) {
        int64_t n = test_sizes[t];
        int64_t valid_len = (n + 7) / 8;

        int32_t *d_group, *h_group;
        int64_t *d_value, *h_value, *d_counts, *h_counts, *d_sums, *h_sums, *h_ref_counts, *h_ref_sums;
        uint8_t *d_valid, *h_valid;

        h_group = (int32_t *)malloc(n * sizeof(int32_t));
        h_value = (int64_t *)malloc(n * sizeof(int64_t));
        h_valid = (uint8_t *)malloc(valid_len);
        h_counts = (int64_t *)malloc(g * sizeof(int64_t));
        h_sums = (int64_t *)malloc(g * sizeof(int64_t));
        h_ref_counts = (int64_t *)malloc(g * sizeof(int64_t));
        h_ref_sums = (int64_t *)malloc(g * sizeof(int64_t));

        memset(h_valid, 0, valid_len);
        for (int64_t i = 0; i < n; i++) {
            h_group[i] = i % g;
            h_value[i] = i * 2;
            if (i % 3 != 0) {
                int64_t byte_idx = i / 8;
                int bit_idx = i % 8;
                h_valid[byte_idx] |= (1 << bit_idx);
            }
        }

        memset(h_counts, 0, g * sizeof(int64_t));
        memset(h_sums, 0, g * sizeof(int64_t));

        cudaMalloc(&d_group, n * sizeof(int32_t));
        cudaMalloc(&d_value, n * sizeof(int64_t));
        cudaMalloc(&d_valid, valid_len);
        cudaMalloc(&d_counts, g * sizeof(int64_t));
        cudaMalloc(&d_sums, g * sizeof(int64_t));

        cudaMemcpy(d_group, h_group, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, h_value, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valid, h_valid, valid_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, h_counts, g * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sums, h_sums, g * sizeof(int64_t), cudaMemcpyHostToDevice);

        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_group, d_value, d_valid, d_counts, d_sums, n, g);
        cudaDeviceSynchronize();

        cudaMemcpy(h_counts, d_counts, g * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sums, d_sums, g * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_group, h_value, h_valid, h_ref_counts, h_ref_sums, n, g);

        int match = 1;
        for (int64_t i = 0; i < g; i++) {
            if (h_counts[i] != h_ref_counts[i] || h_sums[i] != h_ref_sums[i]) {
                match = 0;
                break;
            }
        }

        printf("Test %d (n=%lld): %s\n", t, n, match ? "PASS" : "FAIL");

        free(h_group); free(h_value); free(h_valid);
        free(h_counts); free(h_sums); free(h_ref_counts); free(h_ref_sums);
        cudaFree(d_group); cudaFree(d_value); cudaFree(d_valid);
        cudaFree(d_counts); cudaFree(d_sums);
    }

    return 0;
}
