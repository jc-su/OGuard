#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <algorithm>

extern "C" __global__ void kernel(double* values, double* bounds, int32_t* out, int64_t n, int64_t b);

void test_bucketize(int64_t n, int64_t b) {
    printf("Testing bucketize with n=%ld, b=%ld\n", n, b);

    // Allocate host memory
    double* h_values = new double[n];
    double* h_bounds = new double[b];
    int32_t* h_out = new int32_t[n];
    int32_t* h_expected = new int32_t[n];

    // Initialize bounds
    double bounds_template[] = {10.0, 25.0, 50.0, 75.0, 90.0};
    for (int64_t i = 0; i < b; i++) {
        h_bounds[i] = bounds_template[i % 5];
    }
    std::sort(h_bounds, h_bounds + b);

    // Initialize values
    for (int64_t i = 0; i < n; i++) {
        h_values[i] = fmod(i * 7.5, 100.0);
    }

    // Compute expected output
    for (int64_t i = 0; i < n; i++) {
        double value = h_values[i];
        int64_t count = 0;
        for (int64_t j = 0; j < b; j++) {
            if (h_bounds[j] <= value) {
                count++;
            } else {
                break;
            }
        }
        h_expected[i] = (int32_t)count;
    }

    // Allocate device memory
    double* d_values = nullptr;
    double* d_bounds = nullptr;
    int32_t* d_out = nullptr;
    cudaMalloc(&d_values, n * sizeof(double));
    cudaMalloc(&d_bounds, b * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(d_values, h_values, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bounds, h_bounds, b * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_values, d_bounds, d_out, n, b);

    // Copy result back
    cudaMemcpy(h_out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        if (h_out[i] != h_expected[i]) {
            pass = false;
            if (n <= 10) {
                printf("  mismatch at i=%ld: got %d, expected %d (value=%f)\n", i, h_out[i], h_expected[i], h_values[i]);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_bounds);
    cudaFree(d_out);
    delete[] h_values;
    delete[] h_bounds;
    delete[] h_out;
    delete[] h_expected;
}

int main() {
    int64_t test_configs[][2] = {{0, 5}, {1, 5}, {1000, 3}, {1000003, 10}};
    for (int i = 0; i < 4; i++) {
        test_bucketize(test_configs[i][0], test_configs[i][1]);
    }
    return 0;
}
