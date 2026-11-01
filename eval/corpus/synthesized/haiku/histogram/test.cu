#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

extern "C" __global__ void kernel(
    const uint32_t* keys, int64_t shift,
    int32_t* counts);

int main() {
    int64_t n = 1000000;
    uint32_t* h_keys = new uint32_t[n];
    int32_t* h_counts = new int32_t[256];

    for (int64_t i = 0; i < n; i++) {
        h_keys[i] = i * 31;
    }

    uint32_t* d_keys;
    int32_t* d_counts;

    cudaMalloc(&d_keys, n * sizeof(uint32_t));
    cudaMalloc(&d_counts, 256 * sizeof(int32_t));

    cudaMemcpy(d_keys, h_keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (int shift = 0; shift <= 24; shift += 8) {
        cudaMemset(d_counts, 0, 256 * sizeof(int32_t));

        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        kernel<<<blocks, threads_per_block>>>(d_keys, shift, d_counts);

        cudaDeviceSynchronize();

        cudaMemcpy(h_counts, d_counts, 256 * sizeof(int32_t), cudaMemcpyDeviceToHost);

        int32_t total = 0;
        for (int i = 0; i < 256; i++) {
            total += h_counts[i];
        }
        assert(total == n);

        printf("Test shift=%d passed (total=%d)\n", shift, total);
    }

    cudaFree(d_keys);
    cudaFree(d_counts);

    delete[] h_keys;
    delete[] h_counts;

    printf("All tests passed!\n");
    return 0;
}
