#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(int* values, int* out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long tile_id = idx / 1024;
        long long pos_in_tile = idx % 1024;
        long long tile_start = tile_id * 1024;
        long long tile_end = (tile_start + 1024 < n) ? tile_start + 1024 : n;
        long long tile_len = tile_end - tile_start;
        
        long long rev_pos = tile_len - 1 - pos_in_tile;
        out[idx] = values[tile_start + rev_pos];
    }
}

void cpu_reference(int* values, int* out, long long n) {
    for (long long i = 0; i < n; i++) {
        long long tile_id = i / 1024;
        long long pos_in_tile = i % 1024;
        long long tile_start = tile_id * 1024;
        long long tile_end = (tile_start + 1024 < n) ? tile_start + 1024 : n;
        long long tile_len = tile_end - tile_start;
        long long rev_pos = tile_len - 1 - pos_in_tile;
        out[i] = values[tile_start + rev_pos];
    }
}

bool test_tile_reverse(long long n) {
    int* h_values = (int*)malloc(n * sizeof(int));
    int* h_out = (int*)malloc(n * sizeof(int));
    int* h_ref = (int*)malloc(n * sizeof(int));
    
    for (long long i = 0; i < n; i++) {
        h_values[i] = rand() % 2000 - 1000;
    }
    
    int* d_values;
    int* d_out;
    cudaMalloc(&d_values, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));
    
    cudaMemcpy(d_values, h_values, n * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsperblock = 256;
    int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
    if (n > 0) {
        kernel<<<blockspergrid, threadsperblock>>>(d_values, d_out, n);
    }
    
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    cpu_reference(h_values, h_ref, n);
    
    bool passed = memcmp(h_out, h_ref, n * sizeof(int)) == 0;
    printf("n=%lld: %s\n", n, passed ? "PASS" : "FAIL");
    
    cudaFree(d_values);
    cudaFree(d_out);
    free(h_values);
    free(h_out);
    free(h_ref);
    
    return passed;
}

int main() {
    cudaSetDevice(0);
    
    int passed = 0, total = 0;
    
    for (long long n : {0LL, 1LL, 1024LL, 1025LL, 2048LL, 1000003LL}) {
        total++;
        if (test_tile_reverse(n)) {
            passed++;
        }
    }
    
    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
