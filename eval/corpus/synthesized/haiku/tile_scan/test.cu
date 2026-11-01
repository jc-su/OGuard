#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int32_t* values, int32_t* out, int32_t* totals, int64_t n);

void test_tile_scan(int64_t n) {
    printf("Testing tile_scan with n=%ld\n", n);

    const int TILE_SIZE = 1024;
    int64_t num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate host memory
    int32_t* h_values = new int32_t[n];
    int32_t* h_out = new int32_t[n];
    int32_t* h_totals = new int32_t[num_tiles];
    int32_t* h_expected_out = new int32_t[n];
    int32_t* h_expected_totals = new int32_t[num_tiles];

    // Initialize values
    for (int64_t i = 0; i < n; i++) {
        h_values[i] = i % 10;
    }

    // Compute expected results
    for (int64_t i = 0; i < n; i++) {
        int64_t tile_idx = i / TILE_SIZE;
        int64_t local_idx = i % TILE_SIZE;
        int64_t tile_start = tile_idx * TILE_SIZE;
        int64_t tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;

        // Prefix sum within tile
        int64_t prefix_sum = 0;
        for (int64_t j = tile_start; j < tile_start + local_idx; j++) {
            prefix_sum += h_values[j];
        }
        h_expected_out[i] = (int32_t)prefix_sum;

        // Tile total (computed by all threads of a tile, but value is same)
        if (local_idx == 0) {
            int64_t total = 0;
            for (int64_t j = tile_start; j < tile_end; j++) {
                total += h_values[j];
            }
            h_expected_totals[tile_idx] = (int32_t)total;
        }
    }

    // Allocate device memory
    int32_t* d_values = nullptr;
    int32_t* d_out = nullptr;
    int32_t* d_totals = nullptr;
    cudaMalloc(&d_values, n * sizeof(int32_t));
    cudaMalloc(&d_out, n * sizeof(int32_t));
    cudaMalloc(&d_totals, num_tiles * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(d_values, h_values, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_values, d_out, d_totals, n);

    // Copy results back
    cudaMemcpy(h_out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_totals, d_totals, num_tiles * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Verify results
    bool out_pass = true;
    for (int64_t i = 0; i < n; i++) {
        if (h_out[i] != h_expected_out[i]) {
            out_pass = false;
            if (n <= 10) {
                printf("  out mismatch at i=%ld: got %d, expected %d\n", i, h_out[i], h_expected_out[i]);
            }
        }
    }

    bool totals_pass = true;
    for (int64_t i = 0; i < num_tiles; i++) {
        if (h_totals[i] != h_expected_totals[i]) {
            totals_pass = false;
            if (num_tiles <= 10) {
                printf("  totals mismatch at i=%ld: got %d, expected %d\n", i, h_totals[i], h_expected_totals[i]);
            }
        }
    }

    if (out_pass && totals_pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_out);
    cudaFree(d_totals);
    delete[] h_values;
    delete[] h_out;
    delete[] h_totals;
    delete[] h_expected_out;
    delete[] h_expected_totals;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int i = 0; i < 4; i++) {
        test_tile_scan(test_sizes[i]);
    }
    return 0;
}
