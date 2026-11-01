#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" __global__ void kernel(
    const double* price, const double* discount, int64_t n,
    double* out);

int main() {
    const int64_t test_sizes[] = {0, 1, 1000, 1000003};

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int64_t n = test_sizes[test_idx];

        double* h_price = new double[n];
        double* h_discount = new double[n];
        double* h_out = new double[n];

        for (int64_t i = 0; i < n; i++) {
            h_price[i] = i * 0.5;
            h_discount[i] = 0.1 + (i % 10) * 0.01;
        }

        double* d_price;
        double* d_discount;
        double* d_out;

        cudaMalloc(&d_price, n * sizeof(double));
        cudaMalloc(&d_discount, n * sizeof(double));
        cudaMalloc(&d_out, n * sizeof(double));

        cudaMemcpy(d_price, h_price, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_discount, h_discount, n * sizeof(double), cudaMemcpyHostToDevice);

        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        if (blocks == 0) blocks = 1;
        kernel<<<blocks, threads_per_block>>>(d_price, d_discount, n, d_out);

        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, n * sizeof(double), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < n; i++) {
            double expected = h_price[i] * (1.0 - h_discount[i]);
            assert(fabs(h_out[i] - expected) < 1e-9);
        }

        printf("Test n=%ld passed\n", n);

        cudaFree(d_price);
        cudaFree(d_discount);
        cudaFree(d_out);

        delete[] h_price;
        delete[] h_discount;
        delete[] h_out;
    }

    printf("All tests passed!\n");
    return 0;
}
