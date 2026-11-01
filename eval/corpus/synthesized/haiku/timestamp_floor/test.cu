#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" __global__ void kernel(int64_t *ts, int64_t *out, int64_t n, int64_t unit);

void cpu_ref(int64_t *ts, int64_t *out, int64_t n, int64_t unit) {
    for (int64_t i = 0; i < n; i++) {
        int64_t ts_val = ts[i];
        if (ts_val >= 0) {
            out[i] = (ts_val / unit) * unit;
        } else {
            if (ts_val % unit == 0) {
                out[i] = ts_val;
            } else {
                out[i] = ((ts_val / unit) - 1) * unit;
            }
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int64_t unit = 1000;
    
    for (int t = 0; t < 4; t++) {
        int64_t n = test_sizes[t];
        
        int64_t *d_ts, *h_ts, *d_out, *h_out, *h_ref;
        
        h_ts = (int64_t *)malloc(n * sizeof(int64_t));
        h_out = (int64_t *)malloc(n * sizeof(int64_t));
        h_ref = (int64_t *)malloc(n * sizeof(int64_t));
        
        for (int64_t i = 0; i < n; i++) h_ts[i] = i - n / 2;
        
        cudaMalloc(&d_ts, n * sizeof(int64_t));
        cudaMalloc(&d_out, n * sizeof(int64_t));
        
        cudaMemcpy(d_ts, h_ts, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        
        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_ts, d_out, n, unit);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_ts, h_ref, n, unit);
        
        printf("Test %d (n=%lld): %s\n", t, n, memcmp(h_out, h_ref, n * sizeof(int64_t)) == 0 ? "PASS" : "FAIL");
        
        free(h_ts); free(h_out); free(h_ref);
        cudaFree(d_ts); cudaFree(d_out);
    }
    return 0;
}
