#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int64_t *values, int64_t *out, int8_t *overflow, int64_t n, int64_t shift);

void cpu_ref(int64_t *values, int64_t *out, int8_t *overflow, int64_t n, int64_t shift) {
    for (int64_t i = 0; i < n; i++) {
        int64_t v = values[i];
        if (shift >= 0) {
            int64_t power = 1;
            for (int j = 0; j < shift; j++) power *= 10;
            __int128 result = (__int128)v * power;
            if (result < INT64_MIN || result > INT64_MAX) {
                overflow[i] = 1;
                out[i] = 0;
            } else {
                overflow[i] = 0;
                out[i] = (int64_t)result;
            }
        } else {
            int64_t shift_abs = -shift;
            int64_t power = 1;
            for (int j = 0; j < shift_abs; j++) power *= 10;
            int64_t abs_v = (v < 0) ? -v : v;
            int64_t rounded = (abs_v + power / 2) / power;
            out[i] = (v < 0) ? -rounded : rounded;
            overflow[i] = 0;
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int num_tests = 4;
    
    for (int t = 0; t < num_tests; t++) {
        int64_t n = test_sizes[t];
        int64_t *d_values, *h_values, *d_out, *h_out, *h_ref;
        int8_t *d_overflow, *h_overflow, *h_ref_overflow;
        
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        h_ref = (int64_t *)malloc(n * sizeof(int64_t));
        h_overflow = (int8_t *)malloc(n);
        h_ref_overflow = (int8_t *)malloc(n);
        
        for (int64_t i = 0; i < n; i++) h_values[i] = i - n / 2;
        
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_out, n * sizeof(int64_t));
        cudaMalloc(&d_overflow, n);
        
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        int64_t shift = 2;
        kernel<<<blocks, 256>>>(d_values, d_out, d_overflow, n, shift);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_overflow, d_overflow, n, cudaMemcpyDeviceToHost);
        cpu_ref(h_values, h_ref, h_ref_overflow, n, shift);
        
        int match = memcmp(h_out, h_ref, n * sizeof(int64_t)) == 0 &&
                    memcmp(h_overflow, h_ref_overflow, n) == 0;
        
        printf("Test %d (n=%lld): %s\n", t, n, match ? "PASS" : "FAIL");
        
        free(h_values); free(h_out); free(h_ref);
        free(h_overflow); free(h_ref_overflow);
        cudaFree(d_values); cudaFree(d_out); cudaFree(d_overflow);
    }
    
    return 0;
}
