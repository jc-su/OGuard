#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <limits.h>

__global__ void kernel(int* offsets, long long* values, long long* maxes, int g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < g) {
        int start = offsets[idx];
        int end = offsets[idx + 1];

        // Initialize with smallest int64
        long long max_val = LLONG_MIN;

        // Find max in this segment
        for (int i = start; i < end; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }

        maxes[idx] = max_val;
    }
}

void cpu_reference(int* offsets, long long* values, long long* maxes, int g) {
    for (int i = 0; i < g; i++) {
        int start = offsets[i];
        int end = offsets[i + 1];

        long long max_val = LLONG_MIN;
        for (int j = start; j < end; j++) {
            if (values[j] > max_val) {
                max_val = values[j];
            }
        }
        maxes[i] = max_val;
    }
}

bool test_segment_max(int n) {
    int g = 0;
    int* h_offsets = NULL;
    long long* h_values = NULL;
    long long* h_maxes = NULL;
    long long* h_ref = NULL;

    if (n == 0) {
        g = 0;
        h_offsets = (int*)malloc(sizeof(int));
        h_offsets[0] = 0;
        h_values = (long long*)malloc(sizeof(long long));
        h_maxes = (long long*)malloc(sizeof(long long));
        h_ref = (long long*)malloc(sizeof(long long));
    } else {
        // Create test data
        h_values = (long long*)malloc(n * sizeof(long long));
        for (int i = 0; i < n; i++) {
            h_values[i] = rand() % 2000 - 1000;
        }

        // Create offsets
        g = (n / 10) + 1;
        if (g > 100) g = 100;
        h_offsets = (int*)malloc((g + 1) * sizeof(int));

        // Generate offsets
        for (int i = 0; i <= g; i++) {
            h_offsets[i] = (i * n) / g;
        }
        h_offsets[g] = n;

        h_maxes = (long long*)malloc(g * sizeof(long long));
        h_ref = (long long*)malloc(g * sizeof(long long));
    }

    // Allocate device memory
    int* d_offsets;
    long long* d_values;
    long long* d_maxes;

    if (g > 0 || n == 0) {
        cudaMalloc(&d_offsets, (g + 1) * sizeof(int));
        cudaMalloc(&d_maxes, g * sizeof(long long));
        if (n > 0) {
            cudaMalloc(&d_values, n * sizeof(long long));
            cudaMemcpy(d_values, h_values, n * sizeof(long long), cudaMemcpyHostToDevice);
            cudaMemcpy(d_offsets, h_offsets, (g + 1) * sizeof(int), cudaMemcpyHostToDevice);
        }

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (g + threadsperblock - 1) / threadsperblock;
        if (g > 0) {
            kernel<<<blockspergrid, threadsperblock>>>(d_offsets, d_values, d_maxes, g);
        }

        // Copy result back
        if (g > 0) {
            cudaMemcpy(h_maxes, d_maxes, g * sizeof(long long), cudaMemcpyDeviceToHost);
        }

        // Get CPU reference
        cpu_reference(h_offsets, h_values, h_ref, g);

        // Compare
        bool passed = (g == 0) || (memcmp(h_maxes, h_ref, g * sizeof(long long)) == 0);
        printf("n=%d: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed) {
            printf("First 10 results (expected vs got):\n");
            for (int i = 0; i < (g > 10 ? 10 : g); i++) {
                printf("  [%d] expected %lld, got %lld\n", i, h_ref[i], h_maxes[i]);
            }
        }

        // Cleanup
        if (g > 0) {
            cudaFree(d_offsets);
            cudaFree(d_maxes);
            if (n > 0) {
                cudaFree(d_values);
            }
        }
    } else {
        printf("n=%d: SKIP\n", n);
    }

    free(h_offsets);
    free(h_values);
    free(h_maxes);
    free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (int n : {0, 1, 1000, 1000003}) {
        total++;
        if (test_segment_max(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
