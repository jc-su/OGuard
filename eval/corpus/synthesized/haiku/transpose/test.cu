#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* source,
    int32_t* out,
    int64_t rows,
    int64_t cols
);

void cpu_reference(const int32_t* source, int64_t rows, int64_t cols, int32_t* out_ref) {
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            out_ref[j * rows + i] = source[i * cols + j];
        }
    }
}

void test_transpose(int64_t rows, int64_t cols) {
    int64_t total = rows * cols;

    // Allocate host memory
    int32_t* source_h = new int32_t[total];
    int32_t* out_h = new int32_t[total];
    int32_t* out_ref = new int32_t[total];

    // Initialize data
    for (int64_t i = 0; i < total; i++) {
        source_h[i] = i;
    }

    memset(out_h, 0, total * sizeof(int32_t));

    // Allocate device memory
    int32_t* source_d;
    int32_t* out_d;

    cudaMalloc(&source_d, total * sizeof(int32_t));
    cudaMalloc(&out_d, total * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(source_d, source_h, total * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (total + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(source_d, out_d, rows, cols);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, total * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(source_h, rows, cols, out_ref);

    // Check results
    for (int64_t i = 0; i < total; i++) {
        assert(out_h[i] == out_ref[i]);
    }

    std::cout << "Test passed for rows=" << rows << ", cols=" << cols << std::endl;

    // Cleanup
    cudaFree(source_d);
    cudaFree(out_d);

    delete[] source_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    test_transpose(0, 0);
    test_transpose(1, 1);
    test_transpose(10, 100);
    test_transpose(1000, 1001);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
