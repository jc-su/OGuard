#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *first, int64_t *out, int64_t n);

void cpu_ref(int64_t *values, int32_t *first, int64_t *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int32_t segment_start = first[i];
        int64_t max_val = values[segment_start];
        for (int64_t j = segment_start; j <= i; j++) {
            if (values[j] > max_val) max_val = values[j];
        }
        out[i] = max_val;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_values, *h_values, *d_out, *h_out, *h_ref;
        int32_t *d_first, *h_first;
        
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_first = (int32_t *)malloc(n * sizeof(int32_t));
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        h_ref = (int64_t *)malloc(n * sizeof(int64_t));
        
        for (int64_t i = 0; i < n; i++) {
            h_values[i] = i - n / 2;
            h_first[i] = (i % 10 == 0) ? i : h_first[i - 1];
        }
        
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_first, n * sizeof(int32_t));
        cudaMalloc(&d_out, n * sizeof(int64_t));
        
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_first, h_first, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_values, d_first, d_out, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_values, h_first, h_ref, n);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(int64_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_values); free(h_first); free(h_out); free(h_ref);
        cudaFree(d_values); cudaFree(d_first); cudaFree(d_out);
    }
    return 0;
}
