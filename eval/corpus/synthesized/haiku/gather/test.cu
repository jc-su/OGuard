#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

extern "C" __global__ void kernel(
    const double* col, const int32_t* sel, int64_t m, int64_t n,
    double* out);

int main() {
    const int64_t test_m[] = {0, 1, 1000, 1000003};

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int64_t m = test_m[test_idx];
        int64_t n = 2000000;

        double* h_col = new double[n];
        int32_t* h_sel = new int32_t[m];
        double* h_out = new double[m];

        for (int64_t i = 0; i < n; i++) {
            h_col[i] = i * 0.5;
        }
        for (int64_t i = 0; i < m; i++) {
            h_sel[i] = i % n;
        }

        double* d_col;
        int32_t* d_sel;
        double* d_out;

        cudaMalloc(&d_col, n * sizeof(double));
        cudaMalloc(&d_sel, m * sizeof(int32_t));
        cudaMalloc(&d_out, m * sizeof(double));

        cudaMemcpy(d_col, h_col, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sel, h_sel, m * sizeof(int32_t), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (m + threads_per_block - 1) / threads_per_block;
        if (blocks == 0) blocks = 1;
        kernel<<<blocks, threads_per_block>>>(d_col, d_sel, m, n, d_out);

        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, m * sizeof(double), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < m; i++) {
            assert(h_out[i] == h_col[h_sel[i]]);
        }

        printf("Test m=%ld, n=%ld passed\n", m, n);

        cudaFree(d_col);
        cudaFree(d_sel);
        cudaFree(d_out);

        delete[] h_col;
        delete[] h_sel;
        delete[] h_out;
    }

    printf("All tests passed!\n");
    return 0;
}
