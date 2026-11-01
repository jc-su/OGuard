#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel(int* idx, double* values, double* out, long long m, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int out_idx = idx[i];
        if (out_idx >= 0 && out_idx < n) {
            atomicAdd(&out[out_idx], values[i]);
        }
    }
}

void cpu_reference(int* idx, double* values, double* out, long long m, long long n) {
    for (long long i = 0; i < m; i++) {
        int out_idx = idx[i];
        if (out_idx >= 0 && out_idx < n) {
            out[out_idx] += values[i];
        }
    }
}

bool test_scatter_add(long long m, long long n) {
    int* h_idx = (int*)malloc(m * sizeof(int));
    double* h_values = (double*)malloc(m * sizeof(double));
    double* h_out = (double*)malloc(n * sizeof(double));
    double* h_ref = (double*)malloc(n * sizeof(double));
    
    for (long long i = 0; i < m; i++) {
        h_idx[i] = rand() % n;
        h_values[i] = (double)(rand() % 200 - 100);
    }
    
    memset(h_out, 0, n * sizeof(double));
    memset(h_ref, 0, n * sizeof(double));
    
    int* d_idx;
    double* d_values;
    double* d_out;
    cudaMalloc(&d_idx, m * sizeof(int));
    cudaMalloc(&d_values, m * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    
    cudaMemcpy(d_idx, h_idx, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsperblock = 256;
    int blockspergrid = (m + threadsperblock - 1) / threadsperblock;
    if (m > 0) {
        kernel<<<blockspergrid, threadsperblock>>>(d_idx, d_values, d_out, m, n);
    }
    
    cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    cpu_reference(h_idx, h_values, h_ref, m, n);
    
    bool passed = true;
    for (long long i = 0; i < n; i++) {
        if (fabs(h_out[i] - h_ref[i]) > 1e-10) {
            passed = false;
            break;
        }
    }
    
    printf("m=%lld, n=%lld: %s\n", m, n, passed ? "PASS" : "FAIL");
    
    cudaFree(d_idx);
    cudaFree(d_values);
    cudaFree(d_out);
    free(h_idx);
    free(h_values);
    free(h_out);
    free(h_ref);
    
    return passed;
}

int main() {
    cudaSetDevice(0);
    
    int passed = 0, total = 0;
    
    for (long long m : {0LL, 1LL, 1000LL, 100000LL}) {
        for (long long n : {1LL, 10LL, 100LL}) {
            total++;
            if (test_scatter_add(m, n)) {
                passed++;
            }
        }
    }
    
    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
