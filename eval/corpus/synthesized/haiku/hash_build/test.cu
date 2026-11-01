#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

extern "C" __global__ void kernel(
    const int64_t* keys, int64_t capacity,
    int64_t* table_keys, int32_t* table_rows);

int main() {
    const int64_t test_sizes[] = {0, 1, 100, 10000};

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int64_t n = test_sizes[test_idx];
        int64_t capacity = (n < 2) ? 4 : (1LL << (64 - __builtin_clzll(2*n - 1)));

        int64_t* h_keys = new int64_t[n];
        int64_t* h_table_keys = new int64_t[capacity];
        int32_t* h_table_rows = new int32_t[capacity];

        for (int64_t i = 0; i < n; i++) {
            h_keys[i] = i * 2 + 1;
        }

        for (int64_t i = 0; i < capacity; i++) {
            h_table_keys[i] = -1;
            h_table_rows[i] = 0;
        }

        int64_t* d_keys;
        int64_t* d_table_keys;
        int32_t* d_table_rows;

        cudaMalloc(&d_keys, n * sizeof(int64_t));
        cudaMalloc(&d_table_keys, capacity * sizeof(int64_t));
        cudaMalloc(&d_table_rows, capacity * sizeof(int32_t));

        cudaMemcpy(d_keys, h_keys, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_keys, h_table_keys, capacity * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_rows, h_table_rows, capacity * sizeof(int32_t), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        if (blocks == 0) blocks = 1;
        kernel<<<blocks, threads_per_block>>>(d_keys, capacity, d_table_keys, d_table_rows);

        cudaDeviceSynchronize();

        cudaMemcpy(h_table_keys, d_table_keys, capacity * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_table_rows, d_table_rows, capacity * sizeof(int32_t), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < n; i++) {
            int64_t key = h_keys[i];
            int64_t hash_val = key & (capacity - 1);
            while (h_table_keys[hash_val] != (int64_t)-1) {
                if (h_table_keys[hash_val] == key) {
                    assert(h_table_rows[hash_val] == i);
                    break;
                }
                hash_val = (hash_val + 1) & (capacity - 1);
            }
        }

        printf("Test n=%ld passed\n", n);

        cudaFree(d_keys);
        cudaFree(d_table_keys);
        cudaFree(d_table_rows);

        delete[] h_keys;
        delete[] h_table_keys;
        delete[] h_table_rows;
    }

    printf("All tests passed!\n");
    return 0;
}
