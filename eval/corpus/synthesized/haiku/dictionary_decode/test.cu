#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <assert.h>
#include "kernel.cu"

void cpu_reference(int32_t *codes, double *dictionary, double *out_ref, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        int32_t code = codes[i];
        out_ref[i] = dictionary[code];
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t d = n > 0 ? (n / 10 + 1) : 1;
        if (d > 256) d = 256;
        
        int32_t *h_codes = (int32_t*)malloc(n * sizeof(int32_t));
        double *h_dictionary = (double*)malloc(d * sizeof(double));
        double *h_out = (double*)malloc(n * sizeof(double));
        double *h_out_ref = (double*)malloc(n * sizeof(double));
        
        for (int64_t i = 0; i < n; i++) {
            h_codes[i] = (int32_t)(rand() % d);
        }
        for (int64_t i = 0; i < d; i++) {
            h_dictionary[i] = (double)rand() / RAND_MAX;
        }
        
        int32_t *d_codes;
        double *d_dictionary;
        double *d_out;
        cudaMalloc(&d_codes, n * sizeof(int32_t));
        cudaMalloc(&d_dictionary, d * sizeof(double));
        cudaMalloc(&d_out, n * sizeof(double));
        
        cudaMemcpy(d_codes, h_codes, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dictionary, h_dictionary, d * sizeof(double), cudaMemcpyHostToDevice);
        
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_codes, d_dictionary, d_out, n, d);
        
        cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
        cpu_reference(h_codes, h_dictionary, h_out_ref, n, d);
        
        for (int64_t i = 0; i < n; i++) {
            double diff = fabs(h_out[i] - h_out_ref[i]);
            assert(diff < 1e-10 || diff / fabs(h_out_ref[i]) < 1e-10);
        }
        
        printf("✓ n=%ld, d=%ld: passed\n", n, d);
        
        cudaFree(d_codes);
        cudaFree(d_dictionary);
        cudaFree(d_out);
        free(h_codes);
        free(h_dictionary);
        free(h_out);
        free(h_out_ref);
    }
    
    printf("All tests passed!\n");
    return 0;
}
