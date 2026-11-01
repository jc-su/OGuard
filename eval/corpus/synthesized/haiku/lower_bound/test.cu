#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(long long* keys, long long* sorted_arr, int* out, long long n, long long s) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long key = keys[idx];
        // Count how many elements in sorted < key
        long long count = 0;
        for (long long i = 0; i < s; i++) {
            if (sorted_arr[i] < key) {
                count++;
            } else {
                break;  // Array is sorted
            }
        }
        out[idx] = (int)count;
    }
}

void cpu_reference(long long* keys, long long* sorted_arr, int* out, long long n, long long s) {
    for (long long i = 0; i < n; i++) {
        long long key = keys[i];
        long long count = 0;
        for (long long j = 0; j < s; j++) {
            if (sorted_arr[j] < key) {
                count++;
            } else {
                break;
            }
        }
        out[i] = (int)count;
    }
}

bool test_lower_bound(long long n) {
    long long s = 100;
    long long* h_keys = (long long*)malloc(n * sizeof(long long));
    long long* h_sorted = (long long*)malloc(s * sizeof(long long));
    int* h_out = (int*)malloc(n * sizeof(int));
    int* h_ref = (int*)malloc(n * sizeof(int));

    // Initialize sorted array
    for (long long i = 0; i < s; i++) {
        h_sorted[i] = (i * 20) - 1000;
    }

    // Initialize random keys
    for (long long i = 0; i < n; i++) {
        h_keys[i] = rand() % 2000 - 1000;
    }

    // Allocate device memory
    long long* d_keys;
    long long* d_sorted;
    int* d_out;

    if (n > 0) {
        cudaMalloc(&d_keys, n * sizeof(long long));
        cudaMalloc(&d_sorted, s * sizeof(long long));
        cudaMalloc(&d_out, n * sizeof(int));

        // Copy to device
        cudaMemcpy(d_keys, h_keys, n * sizeof(long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sorted, h_sorted, s * sizeof(long long), cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_keys, d_sorted, d_out, n, s);

        // Copy result back
        cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

        // Get CPU reference
        cpu_reference(h_keys, h_sorted, h_ref, n, s);

        // Compare
        bool passed = memcmp(h_out, h_ref, n * sizeof(int)) == 0;
        printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed) {
            printf("First 10 results (expected vs got):\n");
            for (long long i = 0; i < (n > 10 ? 10 : n); i++) {
                printf("  [%lld] expected %d, got %d\n", i, h_ref[i], h_out[i]);
            }
        }

        // Cleanup
        cudaFree(d_keys);
        cudaFree(d_sorted);
        cudaFree(d_out);
    } else {
        printf("n=%lld: PASS\n", n);
    }

    free(h_keys);
    free(h_sorted);
    free(h_out);
    free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        total++;
        if (test_lower_bound(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
