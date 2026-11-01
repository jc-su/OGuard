#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *run_ends, int64_t *run_values, int32_t *count, int64_t n);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_values, *h_values, *d_run_values, *h_run_values;
        int32_t *d_run_ends, *h_run_ends, *d_count, *h_count;
        
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_run_ends = (int32_t *)malloc(n * sizeof(int32_t));
        h_run_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_count = (int32_t *)malloc(sizeof(int32_t));
        
        for (int64_t i = 0; i < n; i++) h_values[i] = (i / 10) % 5;
        h_count[0] = 0;
        
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_run_ends, n * sizeof(int32_t));
        cudaMalloc(&d_run_values, n * sizeof(int64_t));
        cudaMalloc(&d_count, sizeof(int32_t));
        
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count, h_count, sizeof(int32_t), cudaMemcpyHostToDevice);
        
        kernel<<<1, 1>>>(d_values, d_run_ends, d_run_values, d_count, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        printf("Test %d (n=%lld): PASS (count=%d)\n", t, n, h_count[0]);
        
        free(h_values); free(h_run_ends); free(h_run_values); free(h_count);
        cudaFree(d_values); cudaFree(d_run_ends); cudaFree(d_run_values); cudaFree(d_count);
    }
    return 0;
}
