#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *ts, int64_t *values, int64_t *out, int64_t n, int64_t range_val);

void cpu_ref(int64_t *ts, int64_t *values, int64_t *out, int64_t n, int64_t range_val) {
    for (int64_t i = 0; i < n; i++) {
        int64_t ts_i = ts[i];
        int64_t lower_bound = ts_i - range_val;
        int64_t total = 0;
        for (int64_t j = 0; j < n; j++) {
            if (ts[j] >= lower_bound && ts[j] <= ts_i) {
                total += values[j];
            }
        }
        out[i] = total;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t range_val = 100;
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_ts, *h_ts, *d_values, *h_values, *d_out, *h_out, *h_ref;
        
        h_ts = (int64_t *)malloc(n * sizeof(int64_t));
        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        h_ref = (int64_t *)malloc(n * sizeof(int64_t));
        
        for (int64_t i = 0; i < n; i++) {
            h_ts[i] = i * 10;
            h_values[i] = i + 1;
        }
        
        cudaMalloc(&d_ts, n * sizeof(int64_t));
        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_out, n * sizeof(int64_t));
        
        cudaMemcpy(d_ts, h_ts, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_ts, d_values, d_out, n, range_val);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_ts, h_values, h_ref, n, range_val);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(int64_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_ts); free(h_values); free(h_out); free(h_ref);
        cudaFree(d_ts); cudaFree(d_values); cudaFree(d_out);
    }
    return 0;
}
