#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__ void kernel(double* col, int32_t* idx, double* out, int64_t m, int64_t n);

void test_gather_nullable(int64_t m, int64_t n) {
    printf("Testing gather_nullable with m=%ld, n=%ld\n", m, n);

    // Allocate host memory
    double* h_col = new double[n];
    int32_t* h_idx = new int32_t[m];
    double* h_out = new double[m];

    // Initialize column with test data
    for (int64_t i = 0; i < n; i++) {
        h_col[i] = i * 1.5;
    }

    // Create indices: mix of valid and -1
    for (int64_t i = 0; i < m; i++) {
        if (i % 3 == 0 && i > 0) {
            h_idx[i] = -1;  // null
        } else {
            h_idx[i] = i % n;  // valid index
        }
    }

    // Allocate device memory
    double* d_col = nullptr;
    int32_t* d_idx = nullptr;
    double* d_out = nullptr;
    cudaMalloc(&d_col, n * sizeof(double));
    cudaMalloc(&d_idx, m * sizeof(int32_t));
    cudaMalloc(&d_out, m * sizeof(double));

    // Copy to device
    cudaMemcpy(d_col, h_col, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, m * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (m + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_col, d_idx, d_out, m, n);

    // Copy result back
    cudaMemcpy(h_out, d_out, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < m; i++) {
        double expected;
        if (h_idx[i] == -1) {
            expected = NAN;
        } else {
            expected = h_col[h_idx[i]];
        }

        bool match = false;
        if (std::isnan(expected) && std::isnan(h_out[i])) {
            match = true;
        } else if (!std::isnan(expected) && std::abs(h_out[i] - expected) < 1e-9) {
            match = true;
        }

        if (!match) {
            pass = false;
            if (m <= 10) {
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
    cudaFree(d_col);
    cudaFree(d_idx);
    cudaFree(d_out);
    delete[] h_col;
    delete[] h_idx;
    delete[] h_out;
}

int main() {
    int64_t test_configs[][2] = {{0, 10}, {1, 10}, {1000, 1000}, {1000003, 1000003}};
    for (int i = 0; i < 4; i++) {
        test_gather_nullable(test_configs[i][0], test_configs[i][1]);
    }
    return 0;
}
