#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" __global__ void kernel(int64_t *left, int64_t *right, int32_t *out,
                                   int64_t n, int64_t r, int64_t tolerance);

void cpu_ref(int64_t *left, int64_t *right, int32_t *out, int64_t n, int64_t r, int64_t tolerance) {
    for (int64_t i = 0; i < n; i++) {
        int64_t left_val = left[i];
        int32_t best_idx = -1;

        for (int64_t j = 0; j < r; j++) {
            if (right[j] <= left_val && left_val - right[j] <= tolerance) {
                best_idx = j;
            }
        }

        out[i] = best_idx;
    }
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    int num_tests = 4;
    int64_t r = 100;
    int64_t tolerance = 50;

    for (int t = 0; t < num_tests; t++) {
        int64_t n = test_sizes[t];

        int64_t *d_left, *h_left, *d_right, *h_right;
        int32_t *d_out, *h_out, *h_ref;

        h_left = (int64_t *)malloc(n * sizeof(int64_t));
        h_right = (int64_t *)malloc(r * sizeof(int64_t));
        h_out = (int32_t *)malloc(n * sizeof(int32_t));
        h_ref = (int32_t *)malloc(n * sizeof(int32_t));

        for (int64_t i = 0; i < n; i++) h_left[i] = i;
        for (int64_t i = 0; i < r; i++) h_right[i] = i * 10;

        cudaMalloc(&d_left, n * sizeof(int64_t));
        cudaMalloc(&d_right, r * sizeof(int64_t));
        cudaMalloc(&d_out, n * sizeof(int32_t));

        cudaMemcpy(d_left, h_left, n * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, h_right, r * sizeof(int64_t), cudaMemcpyHostToDevice);

        int blocks = (n + 255) / 256;
        kernel<<<blocks, 256>>>(d_left, d_right, d_out, n, r, tolerance);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cpu_ref(h_left, h_right, h_ref, n, r, tolerance);

        if (memcmp(h_out, h_ref, n * sizeof(int32_t)) == 0) {
            printf("Test %d (n=%lld): PASS\n", t, n);
        } else {
            printf("Test %d (n=%lld): FAIL\n", t, n);
        }

        free(h_left); free(h_right); free(h_out); free(h_ref);
        cudaFree(d_left); cudaFree(d_right); cudaFree(d_out);
    }

    return 0;
}
