#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

extern "C" __global__ void kernel(int32_t* keys, int32_t* out, int64_t n);

void test_block_sort(int64_t n) {
    printf("Testing block_sort with n=%ld\n", n);

    const int TILE_SIZE = 1024;

    // Allocate host memory
    int32_t* h_keys = new int32_t[n];
    int32_t* h_out = new int32_t[n];
    int32_t* h_expected = new int32_t[n];

    // Initialize with pseudo-random data
    for (int64_t i = 0; i < n; i++) {
        h_keys[i] = (int32_t)((i * 13 + 7) % 1000);
    }

    // Compute expected output
    memcpy(h_expected, h_keys, n * sizeof(int32_t));
    for (int64_t tile_idx = 0; tile_idx < (n + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
        int64_t tile_start = tile_idx * TILE_SIZE;
        int64_t tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;
        std::sort(&h_expected[tile_start], &h_expected[tile_end]);
    }

    // Allocate device memory
    int32_t* d_keys = nullptr;
    int32_t* d_out = nullptr;
    cudaMalloc(&d_keys, n * sizeof(int32_t));
    cudaMalloc(&d_out, n * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(d_keys, h_keys, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_keys, d_out, n);

    // Copy results back
    cudaMemcpy(h_out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Verify results
    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        if (h_out[i] != h_expected[i]) {
            pass = false;
            if (n <= 10) {
                printf("  mismatch at i=%ld: got %d, expected %d\n", i, h_out[i], h_expected[i]);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_out);
    delete[] h_keys;
    delete[] h_out;
    delete[] h_expected;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int i = 0; i < 4; i++) {
        test_block_sort(test_sizes[i]);
    }
    return 0;
}
