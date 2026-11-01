#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* offsets,
    const uint8_t* chars,
    uint8_t* out,
    int64_t n,
    int64_t k
);

void cpu_reference(const int32_t* offsets, const uint8_t* chars, int64_t n, int64_t k,
                   uint8_t* out_ref) {
    for (int64_t i = 0; i < n; i++) {
        int32_t start = offsets[i];
        int32_t end = offsets[i + 1];
        int32_t length = (end - start < k) ? (end - start) : k;

        for (int32_t j = 0; j < length; j++) {
            out_ref[i * k + j] = chars[start + j];
        }
        for (int32_t j = length; j < k; j++) {
            out_ref[i * k + j] = 0;
        }
    }
}

void test_string_prefix(int64_t n, int64_t k) {
    // Allocate host memory
    int32_t* offsets_h = new int32_t[n + 1];
    int64_t total_chars = 0;

    // Create variable-length strings
    for (int64_t i = 0; i < n; i++) {
        int32_t str_len = (i * 7 + 3) % 50;
        offsets_h[i] = total_chars;
        total_chars += str_len;
    }
    offsets_h[n] = total_chars;

    uint8_t* chars_h = new uint8_t[total_chars];
    for (int64_t i = 0; i < total_chars; i++) {
        chars_h[i] = (32 + (i * 7 + 5) % 95);
    }

    uint8_t* out_h = new uint8_t[n * k];
    uint8_t* out_ref = new uint8_t[n * k];

    // Allocate device memory
    int32_t* offsets_d;
    uint8_t* chars_d;
    uint8_t* out_d;

    cudaMalloc(&offsets_d, (n + 1) * sizeof(int32_t));
    cudaMalloc(&chars_d, total_chars * sizeof(uint8_t));
    cudaMalloc(&out_d, n * k * sizeof(uint8_t));

    // Copy to device
    cudaMemcpy(offsets_d, offsets_h, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(chars_d, chars_h, total_chars * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int grid_size = (n + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(offsets_d, chars_d, out_d, n, k);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, n * k * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(offsets_h, chars_h, n, k, out_ref);

    // Check results
    for (int64_t i = 0; i < n * k; i++) {
        assert(out_h[i] == out_ref[i]);
    }

    std::cout << "Test passed for n=" << n << ", k=" << k << std::endl;

    // Cleanup
    cudaFree(offsets_d);
    cudaFree(chars_d);
    cudaFree(out_d);

    delete[] offsets_h;
    delete[] chars_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    test_string_prefix(0, 1);
    test_string_prefix(1, 1);
    test_string_prefix(1000, 32);
    test_string_prefix(1000003, 64);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
