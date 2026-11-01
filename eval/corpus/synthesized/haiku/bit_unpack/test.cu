#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(uint32_t *packed, uint32_t *out, int64_t n, int64_t bits);

void cpu_ref(uint32_t *packed, uint32_t *out, int64_t n, int64_t bits) {
    for (int64_t i = 0; i < n; i++) {
        int64_t bit_pos = i * bits;
        uint32_t code = 0;
        for (int64_t b = 0; b < bits; b++) {
            int64_t pos = bit_pos + b;
            int64_t w_idx = pos / 32;
            int bit_idx = pos % 32;
            uint32_t bit = (packed[w_idx] >> bit_idx) & 1;
            code |= (bit << b);
        }
        out[i] = code;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t bits = 8;
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t w = (n * bits + 31) / 32;
        
        uint32_t *d_packed, *h_packed, *d_out, *h_out, *h_ref;
        
        h_packed = (uint32_t *)malloc(w * sizeof(uint32_t));
        h_out = (uint32_t *)malloc(n * sizeof(uint32_t));
        h_ref = (uint32_t *)malloc(n * sizeof(uint32_t));
        
        for (int64_t i = 0; i < w; i++) h_packed[i] = i % 256;
        
        cudaMalloc(&d_packed, w * sizeof(uint32_t));
        cudaMalloc(&d_out, n * sizeof(uint32_t));
        
        cudaMemcpy(d_packed, h_packed, w * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_packed, d_out, n, bits);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_packed, h_ref, n, bits);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(uint32_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_packed); free(h_out); free(h_ref);
        cudaFree(d_packed); cudaFree(d_out);
    }
    return 0;
}
