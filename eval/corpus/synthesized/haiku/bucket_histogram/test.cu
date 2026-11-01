#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(double* values, double* bounds, int* counts, long long n, int b) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = values[idx];
        // Count how many bounds <= val
        int count = 0;
        for (int i = 0; i < b; i++) {
            if (bounds[i] <= val) {
                count++;
            }
        }
        atomicAdd(&counts[count], 1);
    }
}

void cpu_reference(double* values, double* bounds, int* counts, long long n, int b) {
    for (long long i = 0; i < n; i++) {
        double val = values[i];
        int count = 0;
        for (int j = 0; j < b; j++) {
            if (bounds[j] <= val) {
                count++;
            }
        }
        counts[count]++;
    }
}

bool test_bucket_histogram(long long n) {
    int b = 10;
    double* h_values = (double*)malloc(n * sizeof(double));
    double* h_bounds = (double*)malloc(b * sizeof(double));
    int* h_counts = (int*)malloc((b + 1) * sizeof(int));
    int* h_ref = (int*)malloc((b + 1) * sizeof(int));

    // Initialize bounds
    for (int i = 0; i < b; i++) {
        h_bounds[i] = (i * 10.0) / b;
    }

    // Initialize random values
    for (long long i = 0; i < n; i++) {
        h_values[i] = ((double)rand() / RAND_MAX) * 20.0 - 5.0;
    }

    // Initialize counts
    for (int i = 0; i <= b; i++) {
        h_counts[i] = 0;
        h_ref[i] = 0;
    }

    if (n > 0) {
        // Allocate device memory
        double* d_values;
        double* d_bounds;
        int* d_counts;
        cudaMalloc(&d_values, n * sizeof(double));
        cudaMalloc(&d_bounds, b * sizeof(double));
        cudaMalloc(&d_counts, (b + 1) * sizeof(int));

        // Copy to device
        cudaMemcpy(d_values, h_values, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bounds, h_bounds, b * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, h_counts, (b + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_values, d_bounds, d_counts, n, b);

        // Copy result back
        cudaMemcpy(h_counts, d_counts, (b + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        // Get CPU reference
        cpu_reference(h_values, h_bounds, h_ref, n, b);

        // Compare
        bool passed = memcmp(h_counts, h_ref, (b + 1) * sizeof(int)) == 0;
        printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed) {
            printf("First 11 results (expected vs got):\n");
            for (int i = 0; i <= b; i++) {
                printf("  [%d] expected %d, got %d\n", i, h_ref[i], h_counts[i]);
            }
        }

        // Cleanup
        cudaFree(d_values);
        cudaFree(d_bounds);
        cudaFree(d_counts);
    } else {
        printf("n=%lld: PASS\n", n);
    }

    free(h_values);
    free(h_bounds);
    free(h_counts);
    free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_bucket_histogram(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
