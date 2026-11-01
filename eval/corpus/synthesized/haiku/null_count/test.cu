#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const uint8_t* valid,
    int64_t* count,
    int64_t n
);

void cpu_reference(const uint8_t* valid, int64_t n, int64_t* count_ref) {
    *count_ref = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t byte_idx = i / 8;
        int32_t bit_idx = i % 8;
        if ((valid[byte_idx] & (1 << bit_idx)) == 0) {
            (*count_ref)++;
        }
    }
}

void test_null_count(int64_t n) {
    // Allocate host memory
    int64_t bitmap_size = (n + 7) / 8;
    uint8_t* valid_h = new uint8_t[bitmap_size];
    int64_t count_h = 0;
    int64_t count_ref = 0;

    // Initialize bitmap
    for (int64_t i = 0; i < bitmap_size; i++) {
        valid_h[i] = (i * 73 + 41) & 0xFF;
    }

    // Allocate device memory
    uint8_t* valid_d;
    int64_t* count_d;

    cudaMalloc(&valid_d, bitmap_size * sizeof(uint8_t));
    cudaMalloc(&count_d, sizeof(int64_t));

    // Copy to device
    cudaMemcpy(valid_d, valid_h, bitmap_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(count_d, 0, sizeof(int64_t));

    // Launch kernel
    int grid_size = (n + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(valid_d, count_d, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(&count_h, count_d, sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(valid_h, n, &count_ref);

    // Check results
    assert(count_h == count_ref);

    std::cout << "Test passed for n=" << n << ": " << count_h << " nulls" << std::endl;

    // Cleanup
    cudaFree(valid_d);
    cudaFree(count_d);

    delete[] valid_h;
}

int main() {
    test_null_count(0);
    test_null_count(1);
    test_null_count(1000);
    test_null_count(1000003);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
