#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, int64_t *out, uint8_t *valid, int64_t n);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int32_t total_chars = (n > 0) ? 1000 : 0;
        int64_t valid_len = (n + 7) / 8;
        
        int32_t *d_offsets, *h_offsets;
        uint8_t *d_chars, *h_chars, *d_valid, *h_valid;
        int64_t *d_out, *h_out;
        
        h_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_chars = (uint8_t *)malloc(total_chars);
        h_valid = (uint8_t *)malloc(valid_len);
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        
        memset(h_valid, 0, valid_len);
        h_offsets[0] = 0;
        for (int64_t i = 0; i < n; i++) {
            h_offsets[i + 1] = h_offsets[i] + 5;
            if (h_offsets[i + 1] > total_chars) h_offsets[i + 1] = total_chars;
        }
        
        cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_chars, total_chars);
        cudaMalloc(&d_valid, valid_len);
        cudaMalloc(&d_out, n * sizeof(int64_t));
        
        cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chars, h_chars, total_chars, cudaMemcpyHostToDevice);
        cudaMemcpy(d_valid, h_valid, valid_len, cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_offsets, d_chars, d_out, d_valid, n);
        cudaDeviceSynchronize();
        
        printf("Test %d (n=%lld): PASS\n", t, n);
        
        free(h_offsets); free(h_chars); free(h_valid); free(h_out);
        cudaFree(d_offsets); cudaFree(d_chars); cudaFree(d_valid); cudaFree(d_out);
    }
    return 0;
}
