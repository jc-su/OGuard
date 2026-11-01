#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *row, int64_t n);

void cpu_ref(int64_t *values, int32_t *row, int64_t n) {
    if (n == 0) {
        row[0] = -1;
    } else {
        int64_t min_val = values[0];
        int32_t min_row = 0;
        for (int64_t i = 1; i < n; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
                min_row = i;
            }
        }
        row[0] = min_row;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_values, *h_values;
        int32_t *d_row, *h_row, *h_ref;
        
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_row = (int32_t *)malloc(sizeof(int32_t));
        h_ref = (int32_t *)malloc(sizeof(int32_t));
        
        for (int64_t i = 0; i < n; i++) h_values[i] = i - n / 2;
        h_row[0] = -1;
        
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_row, sizeof(int32_t));
        
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row, h_row, sizeof(int32_t), cudaMemcpyHostToDevice);
        
        kernel<<<1, 1>>>(d_values, d_row, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_row, d_row, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_values, h_ref, n);
        
        printf("Test %d (n=%lld): %s (got %d, expected %d)\n", t, n, 
               h_row[0] == h_ref[0] ? "PASS" : "FAIL", h_row[0], h_ref[0]);
        
        free(h_values); free(h_row); free(h_ref);
        cudaFree(d_values); cudaFree(d_row);
    }
    return 0;
}
