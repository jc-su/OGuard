#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "kernel.cu"

void cpu_reference(uint8_t *valid, int32_t *rows_ref, int32_t *count_ref, int64_t n) {
    int32_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t byte_idx = i / 8;
        int64_t bit_idx = i % 8;
        if ((valid[byte_idx] >> bit_idx) & 1) {
            rows_ref[count++] = (int32_t)i;
        }
    }
    *count_ref = count;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};

    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        int64_t bitmap_size = (n + 7) / 8;

        // Allocate host memory
        uint8_t *h_valid = (uint8_t*)malloc(bitmap_size);
        int32_t *h_rows = (int32_t*)malloc(n * sizeof(int32_t));
        int32_t *h_rows_ref = (int32_t*)malloc(n * sizeof(int32_t));
        int32_t h_count = 0, h_count_ref = 0;

        // Initialize test data
        for (int64_t i = 0; i < bitmap_size; i++) {
            h_valid[i] = (uint8_t)(rand() % 256);
        }

        // Allocate device memory
        uint8_t *d_valid;
        int32_t *d_rows;
        int32_t *d_count;
        cudaMalloc(&d_valid, bitmap_size);
        cudaMalloc(&d_rows, n * sizeof(int32_t));
        cudaMalloc(&d_count, sizeof(int32_t));

        // Copy data to device
        cudaMemcpy(d_valid, h_valid, bitmap_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_count, &h_count, sizeof(int32_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        kernel<<<blockspergrid, threadsperblock>>>(d_valid, d_rows, d_count, n);

        // Copy results back
        cudaMemcpy(h_rows, d_rows, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Compute reference
        cpu_reference(h_valid, h_rows_ref, &h_count_ref, n);

        // Verify count
        assert(h_count == h_count_ref);

        // Verify rows (order doesn't matter, so just check sorted order)
        qsort(h_rows, h_count, sizeof(int32_t), [](const void *a, const void *b) {
            return *(int32_t*)a - *(int32_t*)b;
        });
        qsort(h_rows_ref, h_count_ref, sizeof(int32_t), [](const void *a, const void *b) {
            return *(int32_t*)a - *(int32_t*)b;
        });

        for (int32_t i = 0; i < h_count; i++) {
            assert(h_rows[i] == h_rows_ref[i]);
        }

        printf("✓ n=%ld: passed (count=%d)\n", n, h_count);

        // Cleanup
        cudaFree(d_valid);
        cudaFree(d_rows);
        cudaFree(d_count);
        free(h_valid);
        free(h_rows);
        free(h_rows_ref);
    }

    printf("All tests passed!\n");
    return 0;
}
