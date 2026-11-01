#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__ void kernel(double* a, uint8_t* valid, double* b, double* out, int64_t n);

void test_coalesce(int64_t n) {
    printf("Testing coalesce with n=%ld\n", n);

    // Allocate host memory
    double* h_a = new double[n];
    double* h_b = new double[n];
    double* h_out = new double[n];
    int64_t valid_size = (n + 7) / 8;
    uint8_t* h_valid = new uint8_t[valid_size];

    // Initialize with test data
    for (int64_t i = 0; i < n; i++) {
        h_a[i] = i * 1.5;
        h_b[i] = i * 2.5;
    }

    // Set all rows as valid
    for (int64_t i = 0; i < valid_size; i++) {
        h_valid[i] = 0xFF;
    }
    if (n % 8 != 0) {
        h_valid[valid_size - 1] &= (0xFF >> (8 - (n % 8)));
    }

    // Allocate device memory
    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_out = nullptr;
    uint8_t* d_valid = nullptr;
    cudaMalloc(&d_a, n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    cudaMalloc(&d_valid, valid_size * sizeof(uint8_t));

    // Copy to device
    cudaMemcpy(d_a, h_a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid, h_valid, valid_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_a, d_valid, d_b, d_out, n);

    // Copy result back
    cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        // All rows are valid, so out[i] should equal a[i]
        if (std::abs(h_out[i] - h_a[i]) > 1e-9) {
            pass = false;
            if (n <= 10) {
                printf("  Mismatch at i=%ld: out=%f, a=%f\n", i, h_out[i], h_a[i]);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_valid);
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    delete[] h_valid;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int i = 0; i < 4; i++) {
        test_coalesce(test_sizes[i]);
    }
    return 0;
}
