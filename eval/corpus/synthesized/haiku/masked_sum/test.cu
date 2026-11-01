#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel(double* values, char* mask, double* result, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx] != 0) {
            atomicAdd(result, values[idx]);
        }
    }
}

void cpu_reference(double* values, char* mask, double* result, long long n) {
    *result = 0.0;
    for (long long i = 0; i < n; i++) {
        if (mask[i] != 0) {
            *result += values[i];
        }
    }
}

bool test_masked_sum(long long n) {
    double* h_values = (double*)malloc(n * sizeof(double));
    char* h_mask = (char*)malloc(n * sizeof(char));
    double* h_result = (double*)malloc(sizeof(double));
    double* h_ref = (double*)malloc(sizeof(double));

    // Initialize random values
    for (long long i = 0; i < n; i++) {
        h_values[i] = ((double)rand() / RAND_MAX) * 1000.0 - 500.0;
        h_mask[i] = (char)(rand() % 20 - 10);
    }

    *h_result = 0.0;
    *h_ref = 0.0;

    if (n > 0) {
        // Allocate device memory
        double* d_values;
        char* d_mask;
        double* d_result;
        cudaMalloc(&d_values, n * sizeof(double));
        cudaMalloc(&d_mask, n * sizeof(char));
        cudaMalloc(&d_result, sizeof(double));

        // Copy to device
        cudaMemcpy(d_values, h_values, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, h_mask, n * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_values, d_mask, d_result, n);

        // Copy result back
        cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        // Get CPU reference
        cpu_reference(h_values, h_mask, h_ref, n);

        // Compare (with tolerance for floating point)
        bool passed = fabs(*h_result - *h_ref) < 1e-9 * fabs(*h_ref) + 1e-9;
        printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed) {
            printf("Expected: %.15f\n", *h_ref);
            printf("Got: %.15f\n", *h_result);
        }

        // Cleanup
        cudaFree(d_values);
        cudaFree(d_mask);
        cudaFree(d_result);
    } else {
        printf("n=%lld: PASS\n", n);
    }

    free(h_values);
    free(h_mask);
    free(h_result);
    free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_masked_sum(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
