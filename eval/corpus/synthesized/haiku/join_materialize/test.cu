#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int64_t *build, int64_t *probe, int32_t *offsets, 
                                   int32_t *out_build, int32_t *out_probe, int64_t b, int64_t p);

void cpu_ref(int64_t *build, int64_t *probe, int32_t *offsets, 
             int32_t *out_build, int32_t *out_probe, int64_t b, int64_t p) {
    for (int64_t i = 0; i < p; i++) {
        int32_t start_idx = offsets[i];
        int32_t end_idx = offsets[i + 1];
        for (int32_t k = start_idx; k < end_idx; k++) {
            out_build[k] = k - start_idx;
            out_probe[k] = i;
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t b = 100;
    for (int t = 0; t < 4; t++) {
        int64_t p = test_sizes[t];
        int32_t total_matches = (p > 0) ? 1000 : 0;
        
        int64_t *d_build, *h_build, *d_probe, *h_probe;
        int32_t *d_offsets, *h_offsets, *d_out_build, *h_out_build, *d_out_probe, *h_out_probe;
        int32_t *h_ref_build, *h_ref_probe;
        
        h_build = (int64_t *)malloc(b * sizeof(int64_t));
        h_probe = (int64_t *)malloc(p * sizeof(int64_t));
        h_offsets = (int32_t *)malloc((p + 1) * sizeof(int32_t));
        h_out_build = (int32_t *)malloc(total_matches * sizeof(int32_t));
        h_out_probe = (int32_t *)malloc(total_matches * sizeof(int32_t));
        h_ref_build = (int32_t *)malloc(total_matches * sizeof(int32_t));
        h_ref_probe = (int32_t *)malloc(total_matches * sizeof(int32_t));
        
        h_offsets[0] = 0;
        for (int64_t i = 0; i < p; i++) {
            h_offsets[i + 1] = h_offsets[i] + (i % 10 + 1);
        }
        
        cudaMalloc(&d_build, b * sizeof(int64_t));
        cudaMalloc(&d_probe, p * sizeof(int64_t));
        cudaMalloc(&d_offsets, (p + 1) * sizeof(int32_t));
        cudaMalloc(&d_out_build, total_matches * sizeof(int32_t));
        cudaMalloc(&d_out_probe, total_matches * sizeof(int32_t));
        
        cudaMemcpy(d_offsets, h_offsets, (p + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int blocks = (p + 255) / 256;
        kernel<<<blocks, 256>>>(d_build, d_probe, d_offsets, d_out_build, d_out_probe, b, p);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out_build, d_out_build, total_matches * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_probe, d_out_probe, total_matches * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_build, h_probe, h_offsets, h_ref_build, h_ref_probe, b, p);
        
        printf("Test %d (p=%lld): %s\n", t, p, 
               (memcmp(h_out_build, h_ref_build, total_matches * sizeof(int32_t)) == 0 &&
                memcmp(h_out_probe, h_ref_probe, total_matches * sizeof(int32_t)) == 0) ? "PASS" : "FAIL");
        
        free(h_build); free(h_probe); free(h_offsets);
        free(h_out_build); free(h_out_probe); free(h_ref_build); free(h_ref_probe);
        cudaFree(d_build); cudaFree(d_probe); cudaFree(d_offsets);
        cudaFree(d_out_build); cudaFree(d_out_probe);
    }
    return 0;
}
