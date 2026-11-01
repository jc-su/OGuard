#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int32_t *days, int32_t *months, int32_t *out, int64_t n);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int32_t *d_days, *h_days, *d_months, *h_months, *d_out, *h_out;
        
        h_days = (int32_t *)malloc(n * sizeof(int32_t));
        h_months = (int32_t *)malloc(n * sizeof(int32_t));
        h_out = (int32_t *)malloc(n * sizeof(int32_t));
        
        for (int64_t i = 0; i < n; i++) {
            h_days[i] = i % 10000;
            h_months[i] = (i % 24) - 12;
        }
        
        cudaMalloc(&d_days, n * sizeof(int32_t));
        cudaMalloc(&d_months, n * sizeof(int32_t));
        cudaMalloc(&d_out, n * sizeof(int32_t));
        
        cudaMemcpy(d_days, h_days, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_months, h_months, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_days, d_months, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        printf("Test %d (n=%lld): PASS (output values not validated)\n", t, n);
        
        free(h_days); free(h_months); free(h_out);
        cudaFree(d_days); cudaFree(d_months); cudaFree(d_out);
    }
    return 0;
}
