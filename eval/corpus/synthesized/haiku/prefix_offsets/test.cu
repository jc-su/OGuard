#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* lengths,
    int32_t* offsets,
    int64_t n
);

void cpu_reference(const int32_t* lengths, int64_t n, int32_t* offsets_ref) {
    offsets_ref[0] = 0;
    int64_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        offsets_ref[i] = total;
        total += lengths[i];
    }
    offsets_ref[n] = total;
}

void test_prefix_offsets(int64_t n) {
    // Allocate host memory
    int32_t* lengths_h = new int32_t[n];
    int32_t* offsets_h = new int32_t[n + 1];
    int32_t* offsets_ref = new int32_t[n + 1];

    // Initialize data
    for (int64_t i = 0; i < n; i++) {
        lengths_h[i] = (i * 7 + 3) % 1000;
    }

    memset(offsets_h, 0, (n + 1) * sizeof(int32_t));

    // Allocate device memory
    int32_t* lengths_d;
    int32_t* offsets_d;

    cudaMalloc(&lengths_d, n * sizeof(int32_t));
    cudaMalloc(&offsets_d, (n + 1) * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(lengths_d, lengths_h, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(offsets_d, 0, (n + 1) * sizeof(int32_t));

    // Launch kernel
    int grid_size = (n + 1 + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(lengths_d, offsets_d, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(offsets_h, offsets_d, (n + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(lengths_h, n, offsets_ref);

    // Check results
    for (int64_t i = 0; i <= n; i++) {
        assert(offsets_h[i] == offsets_ref[i]);
    }

    std::cout << "Test passed for n=" << n << std::endl;

    // Cleanup
    cudaFree(lengths_d);
    cudaFree(offsets_d);

    delete[] lengths_h;
    delete[] offsets_h;
    delete[] offsets_ref;
}

int main() {
    test_prefix_offsets(0);
    test_prefix_offsets(1);
    test_prefix_offsets(1000);
    test_prefix_offsets(1000003);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
