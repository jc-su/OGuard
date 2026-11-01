#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* keys,
    int32_t* part,
    int32_t* counts,
    int64_t n,
    int64_t p
);

uint64_t hash_func(int64_t x) {
    uint64_t h = ((uint64_t)x * (uint64_t)0x9E3779B97F4A7C15) >> 32;
    return h;
}

void cpu_reference(const int64_t* keys, int64_t n, int64_t p, int32_t* part_ref, int32_t* counts_ref) {
    memset(counts_ref, 0, p * sizeof(int32_t));
    for (int64_t i = 0; i < n; i++) {
        uint64_t h = hash_func(keys[i]);
        int32_t partition = (int32_t)(h % p);
        part_ref[i] = partition;
        counts_ref[partition]++;
    }
}

void test_hash_partition(int64_t n, int64_t p) {
    // Allocate host memory
    int64_t* keys_h = new int64_t[n];
    int32_t* part_h = new int32_t[n];
    int32_t* counts_h = new int32_t[p];
    int32_t* part_ref = new int32_t[n];
    int32_t* counts_ref = new int32_t[p];

    // Initialize data
    for (int64_t i = 0; i < n; i++) {
        keys_h[i] = (int64_t)(i * 12345 + 67890);
    }

    memset(part_h, 0, n * sizeof(int32_t));
    memset(counts_h, 0, p * sizeof(int32_t));

    // Allocate device memory
    int64_t* keys_d;
    int32_t* part_d;
    int32_t* counts_d;

    cudaMalloc(&keys_d, n * sizeof(int64_t));
    cudaMalloc(&part_d, n * sizeof(int32_t));
    cudaMalloc(&counts_d, p * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(keys_d, keys_h, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(part_d, part_h, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(counts_d, 0, p * sizeof(int32_t));

    // Launch kernel
    int grid_size = (n + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(keys_d, part_d, counts_d, n, p);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(part_h, part_d, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts_h, counts_d, p * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(keys_h, n, p, part_ref, counts_ref);

    // Check results
    for (int64_t i = 0; i < n; i++) {
        assert(part_h[i] == part_ref[i]);
    }
    for (int64_t i = 0; i < p; i++) {
        assert(counts_h[i] == counts_ref[i]);
    }

    std::cout << "Test passed for n=" << n << ", p=" << p << std::endl;

    // Cleanup
    cudaFree(keys_d);
    cudaFree(part_d);
    cudaFree(counts_d);

    delete[] keys_h;
    delete[] part_h;
    delete[] counts_h;
    delete[] part_ref;
    delete[] counts_ref;
}

int main() {
    test_hash_partition(0, 1);
    test_hash_partition(1, 16);
    test_hash_partition(1000, 16);
    test_hash_partition(1000003, 256);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
