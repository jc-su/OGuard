#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int64_t *values, uint8_t *valid, uint8_t *out,
                                   int64_t n, int64_t lo, int64_t hi);

void cpu_ref(int64_t *values, uint8_t *valid, uint8_t *out, int64_t n, int64_t lo, int64_t hi) {
    int64_t out_len = (n + 7) / 8;
    memset(out, 0, out_len);
    for (int64_t i = 0; i < n; i++) {
        int64_t byte_idx = i / 8;
        int bit_idx = i % 8;
        uint8_t valid_bit = (valid[byte_idx] >> bit_idx) & 1;
        if (valid_bit && values[i] >= lo && values[i] <= hi) {
            out[byte_idx] |= (1 << bit_idx);
        }
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int num_tests = 4;

    for (int t = 0; t < num_tests; t++) {
        int64_t n = test_sizes[t];
        int64_t out_len = (n + 7) / 8;

        int64_t *d_values, *h_values;
        uint8_t *d_valid, *h_valid, *d_out, *h_out, *h_ref;

        h_values = (int64_t *)malloc(n * sizeof(int64_t));
        h_valid = (uint8_t *)malloc(out_len);
        h_out = (uint8_t *)malloc(out_len);
        h_ref = (uint8_t *)malloc(out_len);

        for (int64_t i = 0; i < n; i++) {
            h_values[i] = i - n/2;
            int64_t byte_idx = i / 8;
            int bit_idx = i % 8;
            if (i % 3 == 0) h_valid[byte_idx] |= (1 << bit_idx);
        }
        memset(h_out, 0, out_len);

        cudaMalloc(&d_values, n * sizeof(int64_t));
        cudaMalloc(&d_valid, out_len);
        cudaMalloc(&d_out, out_len);

        cudaMemcpy(d_values, h_values, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_valid, h_valid, out_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, h_out, out_len, cudaMemcpyHostToDevice);

        int64_t lo = -100, hi = 100;

        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_values, d_valid, d_out, n, lo, hi);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, out_len, cudaMemcpyDeviceToHost);
        cpu_ref(h_values, h_valid, h_ref, n, lo, hi);

        if (memcmp(h_out, h_ref, out_len) == 0) {
            printf("Test %d (n=%lld): PASS\n", t, n);
        } else {
            printf("Test %d (n=%lld): FAIL\n", t, n);
        }

        free(h_values); free(h_valid); free(h_out); free(h_ref);
        cudaFree(d_values); cudaFree(d_valid); cudaFree(d_out);
    }

    return 0;
}
