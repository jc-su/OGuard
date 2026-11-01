#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *values, uint8_t *valid, int64_t *out, int64_t n);

void cpu_ref(int64_t *values, uint8_t *valid, int64_t *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int64_t last_valid = 0;
        int found = 0;
        
        for (int64_t j = 0; j <= i; j++) {
            int64_t byte_idx = j / 8;
            int bit_idx = j % 8;
            int valid_bit = (valid[byte_idx] >> bit_idx) & 1;
            
            if (valid_bit) {
                last_valid = values[j];
                found = 1;
            }
        }
        
        out[i] = found ? last_valid : 0;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t valid_len = (n + 7) / 8;
        
        int64_t *d_values, *h_values, *d_out, *h_out, *h_ref;
        uint8_t *d_valid, *h_valid;
        
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_valid = (uint8_t *)malloc(valid_len);
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        h_ref = (int64_t *)malloc(n * sizeof(int64_t));
        
        memset(h_valid, 0, valid_len);
        for (int64_t i = 0; i < n; i++) {
            h_values[i] = i * 2;
            if (i % 3 == 0) {
                int64_t byte_idx = i / 8;
                int bit_idx = i % 8;
                h_valid[byte_idx] |= (1 << bit_idx);
            }
        }
        
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_valid, valid_len);
        cudaMalloc(&d_out, n * sizeof(int64_t));
        
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valid, h_valid, valid_len, cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_values, d_valid, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_values, h_valid, h_ref, n);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(int64_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_values); free(h_valid); free(h_out); free(h_ref);
        cudaFree(d_values); cudaFree(d_valid); cudaFree(d_out);
    }
    return 0;
}
