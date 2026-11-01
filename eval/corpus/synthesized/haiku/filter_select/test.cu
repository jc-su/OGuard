#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>

extern "C" __global__ void kernel(
    const double* quantity, int64_t n, double threshold,
    int32_t* sel, int32_t* count);

int main() {
    const int64_t test_sizes[] = {0, 1, 1000, 1000003};

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int64_t n = test_sizes[test_idx];

        // Allocate host memory
        double* h_quantity = new double[n];
        int32_t* h_sel = new int32_t[n];
        int32_t* h_count = new int32_t[1];

        // Initialize input
        for (int64_t i = 0; i < n; i++) {
            h_quantity[i] = i * 0.5;
        }
        double threshold = n * 0.25;

        // Allocate device memory
        double* d_quantity;
        int32_t* d_sel;
        int32_t* d_count;

        cudaMalloc(&d_quantity, n * sizeof(double));
        cudaMalloc(&d_sel, n * sizeof(int32_t));
        cudaMalloc(&d_count, sizeof(int32_t));

        // Copy to device
        cudaMemcpy(d_quantity, h_quantity, n * sizeof(double), cudaMemcpyHostToDevice);
        h_count[0] = 0;
        cudaMemcpy(d_count, h_count, sizeof(int32_t), cudaMemcpyHostToDevice);

        // Launch kernel
        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        if (blocks == 0) blocks = 1;
        kernel<<<blocks, threads_per_block>>>(d_quantity, n, threshold, d_sel, d_count);

        cudaDeviceSynchronize();

        // Copy results back
        cudaMemcpy(h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sel, d_sel, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Verify
        int32_t expected_count = 0;
        for (int64_t i = 0; i < n; i++) {
            if (h_quantity[i] < threshold) {
                expected_count++;
            }
        }

        assert(h_count[0] == expected_count);

        // Verify all selected rows actually pass the threshold
        for (int32_t i = 0; i < h_count[0]; i++) {
            int32_t row = h_sel[i];
            assert(h_quantity[row] < threshold);
        }

        printf("Test n=%ld passed (found %d rows)\n", n, h_count[0]);

        // Cleanup
        cudaFree(d_quantity);
        cudaFree(d_sel);
        cudaFree(d_count);

        delete[] h_quantity;
        delete[] h_sel;
        delete[] h_count;
    }

    printf("All tests passed!\n");
    return 0;
}
