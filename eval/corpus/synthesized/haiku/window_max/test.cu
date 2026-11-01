#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* values,
    int64_t* out,
    int64_t n,
    int64_t w
);

void cpu_reference(const int64_t* values, int64_t n, int64_t w, int64_t* out_ref) {
    for (int64_t i = 0; i < n; i++) {
        int64_t start = (i < w - 1) ? 0 : (i - w + 1);
        int64_t end = i + 1;
        int64_t max_val = values[start];
        for (int64_t j = start + 1; j < end; j++) {
            if (values[j] > max_val) {
                max_val = values[j];
            }
        }
        out_ref[i] = max_val;
    }
}

void test_window_max(int64_t n, int64_t w) {
    // Allocate host memory
    int64_t* values_h = new int64_t[n];
    int64_t* out_h = new int64_t[n];
    int64_t* out_ref = new int64_t[n];

    // Initialize data
    for (int64_t i = 0; i < n; i++) {
        values_h[i] = (int64_t)((i * 7 + 3) % 1000 - 500);
    }

    memset(out_h, 0, n * sizeof(int64_t));

    // Allocate device memory
    int64_t* values_d;
    int64_t* out_d;

    cudaMalloc(&values_d, n * sizeof(int64_t));
    cudaMalloc(&out_d, n * sizeof(int64_t));

    // Copy to device
    cudaMemcpy(values_d, values_h, n * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (n + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(values_d, out_d, n, w);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, n * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(values_h, n, w, out_ref);

    // Check results
    for (int64_t i = 0; i < n; i++) {
        assert(out_h[i] == out_ref[i]);
    }

    std::cout << "Test passed for n=" << n << ", w=" << w << std::endl;

    // Cleanup
    cudaFree(values_d);
    cudaFree(out_d);

    delete[] values_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    test_window_max(0, 1);
    test_window_max(1, 1);
    test_window_max(100, 5);
    test_window_max(1000, 32);
    test_window_max(1000003, 256);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
