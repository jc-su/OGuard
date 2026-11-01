#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(uint8_t *source, uint8_t *out, int64_t n, int64_t offset);

void cpu_ref(uint8_t *source, uint8_t *out, int64_t n, int64_t offset) {
    int64_t out_len = (n + 7) / 8;
    memset(out, 0, out_len);
    for (int64_t i = 0; i < n; i++) {
        int64_t src_byte_idx = (offset + i) / 8;
        int src_bit_idx = (offset + i) % 8;
        uint8_t src_bit = (source[src_byte_idx] >> src_bit_idx) & 1;
        
        int64_t out_byte_idx = i / 8;
        int out_bit_idx = i % 8;
        if (src_bit) out[out_byte_idx] |= (1 << out_bit_idx);
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t offset = 5;
        int64_t src_len = ((offset + n + 7) / 8);
        int64_t out_len = (n + 7) / 8;
        
        uint8_t *d_source, *h_source, *d_out, *h_out, *h_ref;
        h_source = (uint8_t *)malloc(src_len);
        h_out = (uint8_t *)malloc(out_len);
        h_ref = (uint8_t *)malloc(out_len);
        
        for (int64_t i = 0; i < src_len; i++) h_source[i] = (i * 7) % 256;
        
        cudaMalloc(&d_source, src_len);
        cudaMalloc(&d_out, out_len);
        
        cudaMemcpy(d_source, h_source, src_len, cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_source, d_out, n, offset);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, out_len, cudaMemcpyDeviceToHost);
        cpu_ref(h_source, h_ref, n, offset);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, out_len) == 0 ? "PASS" : "FAIL");
        
        free(h_source); free(h_out); free(h_ref);
        cudaFree(d_source); cudaFree(d_out);
    }
    return 0;
}
