#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, int32_t *idx,
                                   int32_t *out_offsets, uint8_t *out_chars,
                                   int64_t n, int64_t m);

void cpu_ref(int32_t *offsets, uint8_t *chars, int32_t *idx,
             int32_t *out_offsets, uint8_t *out_chars, int64_t n, int64_t m) {
    for (int64_t i = 0; i < m; i++) {
        int32_t src_idx = idx[i];
        int32_t src_start = offsets[src_idx];
        int32_t src_end = offsets[src_idx + 1];
        int32_t dst_start = out_offsets[i];

        int32_t length = src_end - src_start;
        for (int32_t j = 0; j < length; j++) {
            out_chars[dst_start + j] = chars[src_start + j];
        }
    }
}

int main() {
    int64_t test_cases[] = {0, 1, 1000, 1000003};
    int num_tests = 4;

    for (int t = 0; t < num_tests; t++) {
        int64_t n = test_cases[t];
        int64_t m = (n > 0) ? n / 2 : 0;

        int32_t *d_offsets, *h_offsets, *d_idx, *h_idx, *d_out_offsets, *h_out_offsets;
        uint8_t *d_chars, *h_chars, *d_out_chars, *h_out_chars, *h_ref;

        int32_t total_chars = (n > 0) ? 1000 : 0;

        h_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_idx = (int32_t *)malloc(m * sizeof(int32_t));
        h_out_offsets = (int32_t *)malloc((m + 1) * sizeof(int32_t));
        h_chars = (uint8_t *)malloc(total_chars);
        h_out_chars = (uint8_t *)malloc(total_chars);
        h_ref = (uint8_t *)malloc(total_chars);

        if (n > 0) {
            h_offsets[0] = 0;
            for (int64_t i = 0; i < n; i++) {
                h_offsets[i + 1] = h_offsets[i] + (i % 10 + 1);
                if (h_offsets[i + 1] > total_chars) h_offsets[i + 1] = total_chars;
            }
            for (int64_t i = 0; i < m; i++) {
                h_idx[i] = i;
            }
            h_out_offsets[0] = 0;
            for (int64_t i = 0; i < m; i++) {
                h_out_offsets[i + 1] = h_out_offsets[i] + (h_offsets[h_idx[i] + 1] - h_offsets[h_idx[i]]);
            }
        } else {
            h_offsets[0] = 0;
        }

        for (int32_t i = 0; i < total_chars; i++) h_chars[i] = i % 256;

        cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_idx, m * sizeof(int32_t));
        cudaMalloc(&d_out_offsets, (m + 1) * sizeof(int32_t));
        cudaMalloc(&d_chars, total_chars);
        cudaMalloc(&d_out_chars, total_chars);

        cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_idx, h_idx, m * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_offsets, h_out_offsets, (m + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chars, h_chars, total_chars, cudaMemcpyHostToDevice);

        int blocks = (m + 255) / 256;
        kernel<<<blocks, 256>>>(d_offsets, d_chars, d_idx, d_out_offsets, d_out_chars, n, m);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out_chars, d_out_chars, total_chars, cudaMemcpyDeviceToHost);
        cpu_ref(h_offsets, h_chars, h_idx, h_out_offsets, h_ref, n, m);

        int32_t out_len = (m > 0) ? h_out_offsets[m] : 0;
        if (memcmp(h_out_chars, h_ref, out_len) == 0) {
            printf("Test %d (n=%lld, m=%lld): PASS\n", t, n, m);
        } else {
            printf("Test %d (n=%lld, m=%lld): FAIL\n", t, n, m);
        }

        free(h_offsets); free(h_idx); free(h_out_offsets);
        free(h_chars); free(h_out_chars); free(h_ref);
        cudaFree(d_offsets); cudaFree(d_idx); cudaFree(d_out_offsets);
        cudaFree(d_chars); cudaFree(d_out_chars);
    }

    return 0;
}
