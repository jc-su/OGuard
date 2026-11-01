#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(long long* values, char* flags, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            flags[idx] = 1;
        } else {
            flags[idx] = (values[idx] != values[idx - 1]) ? 1 : 0;
        }
    }
}

void cpu_reference(long long* values, char* flags, long long n) {
    for (long long i = 0; i < n; i++) {
        if (i == 0 || values[i] != values[i - 1]) {
            flags[i] = 1;
        } else {
            flags[i] = 0;
        }
    }
}

bool test_run_starts(long long n) {
    long long* h_values = (long long*)malloc(n * sizeof(long long));
    char* h_flags = (char*)malloc(n * sizeof(char));
    char* h_ref = (char*)malloc(n * sizeof(char));

    // Initialize random values
    for (long long i = 0; i < n; i++) {
        h_values[i] = rand() % 2000 - 1000;
    }

    // Allocate device memory
    long long* d_values;
    char* d_flags;
    cudaMalloc(&d_values, n * sizeof(long long));
    cudaMalloc(&d_flags, n * sizeof(char));

    // Copy to device
    cudaMemcpy(d_values, h_values, n * sizeof(long long), cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threadsperblock = 256;
    int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
    kernel<<<blockspergrid, threadsperblock>>>(d_values, d_flags, n);

    // Copy result back
    cudaMemcpy(h_flags, d_flags, n * sizeof(char), cudaMemcpyDeviceToHost);

    // Get CPU reference
    cpu_reference(h_values, h_ref, n);

    // Compare
    bool passed = memcmp(h_flags, h_ref, n * sizeof(char)) == 0;
    printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");

    if (!passed) {
        printf("First 10 results (expected vs got):\n");
        for (long long i = 0; i < (n > 10 ? 10 : n); i++) {
            printf("  [%lld] expected %d, got %d\n", i, h_ref[i], h_flags[i]);
        }
    }

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_flags);
    free(h_values);
    free(h_flags);
    free(h_ref);

    return passed;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_run_starts(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
