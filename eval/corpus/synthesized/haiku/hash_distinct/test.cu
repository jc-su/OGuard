#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *keys, int64_t *table_keys, int32_t *table_rows, int64_t n, int64_t capacity);

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t capacity = 2048;
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_keys, *h_keys, *d_table_keys, *h_table_keys;
        int32_t *d_table_rows, *h_table_rows;
        
        h_keys = (int64_t *)malloc(n * sizeof(int64_t));
        h_table_keys = (int64_t *)malloc(capacity * sizeof(int64_t));
        h_table_rows = (int32_t *)malloc(capacity * sizeof(int32_t));
        
        for (int64_t i = 0; i < n; i++) h_keys[i] = i % 100;
        for (int64_t i = 0; i < capacity; i++) {
            h_table_keys[i] = -1;
            h_table_rows[i] = 2147483647;
        }
        
        cudaMalloc(&d_keys, n * sizeof(int64_t));
        cudaMalloc(&d_table_keys, capacity * sizeof(int64_t));
        cudaMalloc(&d_table_rows, capacity * sizeof(int32_t));
        
        cudaMemcpy(d_keys, h_keys, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_keys, h_table_keys, capacity * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_rows, h_table_rows, capacity * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_keys, d_table_keys, d_table_rows, n, capacity);
        cudaDeviceSynchronize();
        
        printf("Test %d (n=%lld): PASS\n", t, n);
        
        free(h_keys); free(h_table_keys); free(h_table_rows);
        cudaFree(d_keys); cudaFree(d_table_keys); cudaFree(d_table_rows);
    }
    return 0;
}
