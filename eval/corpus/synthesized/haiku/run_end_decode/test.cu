#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>
#include <assert.h>
#include "kernel.cu"

void cpu_reference(double *run_values, int32_t *run_ends, double *out_ref, int64_t r, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int64_t left = 0, right = r - 1;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            if (run_ends[mid] <= i) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        out_ref[i] = run_values[left];
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t r = n > 0 ? (n / 100 + 1) : 1;
        
        double *h_run_values = (double*)malloc(r * sizeof(double));
        int32_t *h_run_ends = (int32_t*)malloc(r * sizeof(int32_t));
        double *h_out = (double*)malloc(n * sizeof(double));
        double *h_out_ref = (double*)malloc(n * sizeof(double));
        
        for (int64_t i = 0; i < r; i++) {
            h_run_values[i] = (double)rand() / RAND_MAX;
        }
        
        if (n == 0) {
            h_run_ends[0] = 0;
        } else {
            for (int64_t i = 0; i < r - 1; i++) {
                h_run_ends[i] = 1 + (rand() % (n - 1));
            }
            for (int64_t i = 0; i < r - 1; i++) {
                for (int64_t j = i + 1; j < r - 1; j++) {
                    if (h_run_ends[i] >= h_run_ends[j]) {
                        int32_t tmp = h_run_ends[i];
                        h_run_ends[i] = h_run_ends[j];
                        h_run_ends[j] = tmp;
                    }
                }
            }
            h_run_ends[r - 1] = (int32_t)n;
        }
        
        double *d_run_values;
        int32_t *d_run_ends;
        double *d_out;
        cudaMalloc(&d_run_values, r * sizeof(double));
        cudaMalloc(&d_run_ends, r * sizeof(int32_t));
        cudaMalloc(&d_out, n * sizeof(double));
        
        cudaMemcpy(d_run_values, h_run_values, r * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_run_ends, h_run_ends, r * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_run_values, d_run_ends, d_out, r, n);
        
        cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
        cpu_reference(h_run_values, h_run_ends, h_out_ref, r, n);
        
        for (int64_t i = 0; i < n; i++) {
            double diff = fabs(h_out[i] - h_out_ref[i]);
            assert(diff < 1e-10 || diff / fabs(h_out_ref[i]) < 1e-10);
        }
        
        printf("✓ n=%ld, r=%ld: passed\n", n, r);
        
        cudaFree(d_run_values);
        cudaFree(d_run_ends);
        cudaFree(d_out);
        free(h_run_values);
        free(h_run_ends);
        free(h_out);
        free(h_out_ref);
    }
    
    printf("All tests passed!\n");
    return 0;
}
