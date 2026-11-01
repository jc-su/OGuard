#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <limits.h>

__global__ void kernel(int* group, long long* value, long long* best, long long n, int capacity) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int g = group[idx];
        long long v = value[idx];
        if (g >= 0 && g < capacity) {
            // Atomic max
            atomicMax(&best[g], v);
        }
    }
}

void cpu_reference(int* group, long long* value, long long* best, long long n, int capacity) {
    for (long long i = 0; i < n; i++) {
        int g = group[i];
        if (g >= 0 && g < capacity) {
            if (value[i] > best[g]) {
                best[g] = value[i];
            }
        }
    }
}

bool test_group_max(long long n) {
    int capacity = 100;
    int* h_group = NULL;
    long long* h_value = NULL;
    long long* h_best = NULL;
    long long* h_ref = NULL;

    if (n > 0) {
        h_group = (int*)malloc(n * sizeof(int));
        h_value = (long long*)malloc(n * sizeof(long long));

        // Initialize random values
        for (long long i = 0; i < n; i++) {
            h_group[i] = rand() % capacity;
            h_value[i] = rand() % 2000 - 1000;
        }
    }

    h_best = (long long*)malloc(capacity * sizeof(long long));
    h_ref = (long long*)malloc(capacity * sizeof(long long));

    // Initialize with smallest int64
    for (int i = 0; i < capacity; i++) {
        h_best[i] = LLONG_MIN;
        h_ref[i] = LLONG_MIN;
    }

    if (n > 0) {
        // Allocate device memory
        int* d_group;
        long long* d_value;
        long long* d_best;
        cudaMalloc(&d_group, n * sizeof(int));
        cudaMalloc(&d_value, n * sizeof(long long));
        cudaMalloc(&d_best, capacity * sizeof(long long));

        // Copy to device
        cudaMemcpy(d_group, h_group, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, h_value, n * sizeof(long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_best, h_best, capacity * sizeof(long long), cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_group, d_value, d_best, n, capacity);

        // Copy result back
        cudaMemcpy(h_best, d_best, capacity * sizeof(long long), cudaMemcpyDeviceToHost);

        // Get CPU reference
        cpu_reference(h_group, h_value, h_ref, n, capacity);

        // Compare
        bool passed = memcmp(h_best, h_ref, capacity * sizeof(long long)) == 0;
        printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed) {
            printf("First 10 results (expected vs got):\n");
            for (int i = 0; i < (capacity > 10 ? 10 : capacity); i++) {
                printf("  [%d] expected %lld, got %lld\n", i, h_ref[i], h_best[i]);
            }
        }

        // Cleanup
        cudaFree(d_group);
        cudaFree(d_value);
        cudaFree(d_best);
    } else {
        printf("n=%lld: PASS\n", n);
    }

    if (h_group) free(h_group);
    if (h_value) free(h_value);
    free(h_best);
    free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_group_max(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
