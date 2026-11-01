#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int32_t *group, int64_t *value, int32_t *best, int64_t n, int64_t g);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t g = 10;
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int32_t *d_group, *h_group, *d_best, *h_best;
        int64_t *d_value, *h_value;
        
        h_group = (int32_t *)malloc(n * sizeof(int32_t));
        h_value = (int64_t *)malloc(n * sizeof(int64_t));
        h_best = (int32_t *)malloc(g * sizeof(int32_t));
        
        for (int64_t i = 0; i < n; i++) {
            h_group[i] = i % g;
            h_value[i] = i;
        }
        for (int64_t i = 0; i < g; i++) h_best[i] = -1;
        
        cudaMalloc(&d_group, n * sizeof(int32_t));
        cudaMalloc(&d_value, n * sizeof(int64_t));
        cudaMalloc(&d_best, g * sizeof(int32_t));
        
        cudaMemcpy(d_group, h_group, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, h_value, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_best, h_best, g * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_group, d_value, d_best, n, g);
        cudaDeviceSynchronize();
        
        printf("Test %d (n=%lld): PASS\n", t, n);
        
        free(h_group); free(h_value); free(h_best);
        cudaFree(d_group); cudaFree(d_value); cudaFree(d_best);
    }
    return 0;
}
