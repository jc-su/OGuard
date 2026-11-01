#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* left_idx,
    const int32_t* right_idx,
    const double* left,
    const double* right,
    double* out,
    int64_t m,
    int64_t l,
    int64_t r
);

void cpu_reference(const int32_t* left_idx, const int32_t* right_idx,
                   const double* left, const double* right, int64_t m,
                   double* out_ref) {
    for (int64_t i = 0; i < m; i++) {
        out_ref[i] = left[left_idx[i]] + right[right_idx[i]];
    }
}

void test_pair_sum(int64_t m, int64_t l, int64_t r) {
    // Allocate host memory
    int32_t* left_idx_h = new int32_t[m];
    int32_t* right_idx_h = new int32_t[m];
    double* left_h = new double[l];
    double* right_h = new double[r];
    double* out_h = new double[m];
    double* out_ref = new double[m];

    // Initialize data
    for (int64_t i = 0; i < m; i++) {
        left_idx_h[i] = (int32_t)(i % l);
        right_idx_h[i] = (int32_t)(i % r);
    }

    for (int64_t i = 0; i < l; i++) {
        left_h[i] = (double)i / 10.0;
    }

    for (int64_t i = 0; i < r; i++) {
        right_h[i] = (double)i / 20.0;
    }

    memset(out_h, 0, m * sizeof(double));

    // Allocate device memory
    int32_t* left_idx_d;
    int32_t* right_idx_d;
    double* left_d;
    double* right_d;
    double* out_d;

    cudaMalloc(&left_idx_d, m * sizeof(int32_t));
    cudaMalloc(&right_idx_d, m * sizeof(int32_t));
    cudaMalloc(&left_d, l * sizeof(double));
    cudaMalloc(&right_d, r * sizeof(double));
    cudaMalloc(&out_d, m * sizeof(double));

    // Copy to device
    cudaMemcpy(left_idx_d, left_idx_h, m * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(right_idx_d, right_idx_h, m * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(left_d, left_h, l * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(right_d, right_h, r * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (m + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(left_idx_d, right_idx_d, left_d, right_d, out_d, m, l, r);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(left_idx_h, right_idx_h, left_h, right_h, m, out_ref);

    // Check results (using approximate comparison)
    for (int64_t i = 0; i < m; i++) {
        assert(fabs(out_h[i] - out_ref[i]) < 1e-10);
    }

    std::cout << "Test passed for m=" << m << ", l=" << l << ", r=" << r << std::endl;

    // Cleanup
    cudaFree(left_idx_d);
    cudaFree(right_idx_d);
    cudaFree(left_d);
    cudaFree(right_d);
    cudaFree(out_d);

    delete[] left_idx_h;
    delete[] right_idx_h;
    delete[] left_h;
    delete[] right_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    test_pair_sum(1, 1, 1);
    test_pair_sum(100, 50, 50);
    test_pair_sum(1000, 100, 100);
    test_pair_sum(1000003, 10000, 10000);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
