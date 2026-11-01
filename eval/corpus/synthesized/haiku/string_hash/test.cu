#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int32_t *offsets, uint8_t *chars, uint32_t *out, int64_t n);

void cpu_ref(int32_t *offsets, uint8_t *chars, uint32_t *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        
        uint32_t h = 2166136261U;
        for (int32_t j = start; j < end; j++) {
            uint8_t c = chars[j];
            h = (h ^ c) * 16777619U;
        }
        
        out[i] = h;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int32_t total_chars = (n > 0) ? 1000 : 0;
        
        int32_t *d_offsets, *h_offsets;
        uint8_t *d_chars, *h_chars;
        uint32_t *d_out, *h_out, *h_ref;
        
        h_offsets = (int32_t *)malloc((n + 1) * sizeof(int32_t));
        h_chars = (uint8_t *)malloc(total_chars);
        h_out = (uint32_t *)malloc(n * sizeof(uint32_t));
        h_ref = (uint32_t *)malloc(n * sizeof(uint32_t));
        
        if (n > 0) {
            h_offsets[0] = 0;
            for (int64_t i = 0; i < n; i++) {
                h_offsets[i + 1] = h_offsets[i] + (i % 10 + 1);
                if (h_offsets[i + 1] > total_chars) h_offsets[i + 1] = total_chars;
            }
        }
        for (int32_t i = 0; i < total_chars; i++) h_chars[i] = i % 256;
        
        cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t));
        cudaMalloc(&d_chars, total_chars);
        cudaMalloc(&d_out, n * sizeof(uint32_t));
        
        cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chars, h_chars, total_chars, cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_offsets, d_chars, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_offsets, h_chars, h_ref, n);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(uint32_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_offsets); free(h_chars); free(h_out); free(h_ref);
        cudaFree(d_offsets); cudaFree(d_chars); cudaFree(d_out);
    }
    return 0;
}
