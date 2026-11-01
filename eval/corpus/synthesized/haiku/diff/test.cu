#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(long long* values, long long* out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            out[idx] = values[idx];
        } else {
            out[idx] = values[idx] - values[idx - 1];
        }
    }
}

void cpu_reference(long long* values, long long* out, long long n) {
    if (n > 0) {
        out[0] = values[0];
        for (long long i = 1; i < n; i++) {
            out[i] = values[i] - values[i - 1];
        }
    }
}

bool test_diff(long long n) {
    long long* h_values = (long long*)malloc(n * sizeof(long long));
    long long* h_out = (long long*)malloc(n * sizeof(long long));
    long long* h_ref = (long long*)malloc(n * sizeof(long long));
    
    for (long long i = 0; i < n; i++) {
        h_values[i] = rand() % 2000 - 1000;
    }
    
    long long* d_values;
    long long* d_out;
    cudaMalloc(&d_values, n * sizeof(long long));
    cudaMalloc(&d_out, n * sizeof(long long));
    
    cudaMemcpy(d_values, h_values, n * sizeof(long long), cudaMemcpyHostToDevice);
    
    int threadsperblock = 256;
    int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
    if (n > 0) {
        kernel<<<blockspergrid, threadsperblock>>>(d_values, d_out, n);
    }
    
    cudaMemcpy(h_out, d_out, n * sizeof(long long), cudaMemcpyDeviceToHost);
    
    cpu_reference(h_values, h_ref, n);
    
    bool passed = memcmp(h_out, h_ref, n * sizeof(long long)) == 0;
    printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");
    
    cudaFree(d_values);
    cudaFree(d_out);
    free(h_values);
    free(h_out);
    free(h_ref);
    
    return passed;
}

int main() {
    cudaSetDevice(0);
    
    int passed = 0, total = 0;
    
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_diff(n)) {
            passed++;
        }
    }
    
    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
