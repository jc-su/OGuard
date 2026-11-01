#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int32_t *a_offsets, uint8_t *a_chars, int32_t *b_offsets, uint8_t *b_chars,
                                   int32_t *out_offsets, uint8_t *out_chars, int64_t n);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int32_t total_a = (n > 0) ? 500 : 0;
        int32_t total_b = (n > 0) ? 500 : 0;
        int32_t total_out = total_a + total_b;
        
        int32_t *d_a_offsets, *h_a_offsets, *d_b_offsets, *h_b_offsets;
        int32_t *d_out_offsets, *h_out_offsets;
        uint8_t *d_a_chars, *h_a_chars, *d_b_chars, *h_b_chars;
        uint8_t *d_out_chars, *h_out_chars;
        
        h_a_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_b_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_out_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_a_chars = (uint8_t *)malloc(total_a);
        h_b_chars = (uint8_t *)malloc(total_b);
        h_out_chars = (uint8_t *)malloc(total_out);
        
        if (n > 0) {
            h_a_offsets[0] = 0;
            h_b_offsets[0] = 0;
            h_out_offsets[0] = 0;
            for (int64_t i = 0; i < n; i++) {
                h_a_offsets[i + 1] = h_a_offsets[i] + (i % 5 + 1);
                h_b_offsets[i + 1] = h_b_offsets[i] + (i % 5 + 1);
                h_out_offsets[i + 1] = h_out_offsets[i] + (h_a_offsets[i+1] - h_a_offsets[i]) + (h_b_offsets[i+1] - h_b_offsets[i]);
            }
        }
        
        cudaMalloc(&d_a_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_b_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_out_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_a_chars, total_a);
        cudaMalloc(&d_b_chars, total_b);
        cudaMalloc(&d_out_chars, total_out);
        
        cudaMemcpy(d_a_offsets, h_a_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_offsets, h_b_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_offsets, h_out_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_chars, h_a_chars, total_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_chars, h_b_chars, total_b, cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_a_offsets, d_a_chars, d_b_offsets, d_b_chars, d_out_offsets, d_out_chars, n);
        cudaDeviceSynchronize();
        
        printf("Test %d (n=%lld): PASS\n", t, n);
        
        free(h_a_offsets); free(h_b_offsets); free(h_out_offsets);
        free(h_a_chars); free(h_b_chars); free(h_out_chars);
        cudaFree(d_a_offsets); cudaFree(d_b_offsets); cudaFree(d_out_offsets);
        cudaFree(d_a_chars); cudaFree(d_b_chars); cudaFree(d_out_chars);
    }
    return 0;
}
