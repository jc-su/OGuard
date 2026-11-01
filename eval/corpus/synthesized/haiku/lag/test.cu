#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__ void kernel(double* values, int32_t* first, double* out, int64_t n, int64_t k);

void test_lag(int64_t n, int64_t k) {
    printf("Testing lag with n=%ld, k=%ld\n", n, k);

    // Allocate host memory
    double* h_values = new double[n];
    int32_t* h_first = new int32_t[n];
    double* h_out = new double[n];

    // Initialize values
    for (int64_t i = 0; i < n; i++) {
        h_values[i] = i * 1.5;
    }

    // Create partitions
    for (int64_t i = 0; i < n; i++) {
        if (i < n / 2) {
            h_first[i] = 0;
        } else {
            h_first[i] = n / 2;
        }
    }

    // Allocate device memory
    double* d_values = nullptr;
    int32_t* d_first = nullptr;
    double* d_out = nullptr;
    cudaMalloc(&d_values, n * sizeof(double));
    cudaMalloc(&d_first, n * sizeof(int32_t));
    cudaMalloc(&d_out, n * sizeof(double));

    // Copy to device
    cudaMemcpy(d_values, h_values, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_first, h_first, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_values, d_first, d_out, n, k);

    // Copy result back
    cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        int64_t lag_idx = i - k;
        int32_t first_row = h_first[i];

        double expected;
        if (lag_idx >= first_row) {
            expected = h_values[lag_idx];
        } else {
            expected = NAN;
        }

        bool match = false;
        if (std::isnan(expected) && std::isnan(h_out[i])) {
            match = true;
        } else if (!std::isnan(expected) && std::abs(h_out[i] - expected) < 1e-9) {
            match = true;
        }

        if (!match) {
            pass = false;
            if (n <= 10) {
                printf("  Mismatch at i=%ld: out=%f, expected=%f\n", i, h_out[i], expected);
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
    cudaFree(d_first);
    cudaFree(d_out);
    delete[] h_values;
    delete[] h_first;
    delete[] h_out;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t k = 2;
    for (int i = 0; i < 4; i++) {
        test_lag(test_sizes[i], k);
    }
    return 0;
}
