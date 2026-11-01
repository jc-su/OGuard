#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" __global__ void kernel(
    const int32_t* shipdate, const double* discount, const double* quantity, const double* price,
    int32_t date_lo, int32_t date_hi, double disc_lo, double disc_hi, double qty_max,
    double* result);

int main() {
    const int64_t test_sizes[] = {0, 1, 1000, 1000003};

    for (int test_idx = 0; test_idx < 4; test_idx++) {
        int64_t n = test_sizes[test_idx];

        int32_t* h_shipdate = new int32_t[n];
        double* h_discount = new double[n];
        double* h_quantity = new double[n];
        double* h_price = new double[n];
        double* h_result = new double[1];

        for (int64_t i = 0; i < n; i++) {
            h_shipdate[i] = i % 20000;
            h_discount[i] = 0.05 + (i % 10) * 0.01;
            h_quantity[i] = i * 0.5;
            h_price[i] = i * 0.1;
        }
        h_result[0] = 0.0;

        int32_t* d_shipdate;
        double* d_discount;
        double* d_quantity;
        double* d_price;
        double* d_result;

        cudaMalloc(&d_shipdate, n * sizeof(int32_t));
        cudaMalloc(&d_discount, n * sizeof(double));
        cudaMalloc(&d_quantity, n * sizeof(double));
        cudaMalloc(&d_price, n * sizeof(double));
        cudaMalloc(&d_result, sizeof(double));

        cudaMemcpy(d_shipdate, h_shipdate, n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_discount, h_discount, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_quantity, h_quantity, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_price, h_price, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);

        int32_t date_lo = 5000, date_hi = 15000;
        double disc_lo = 0.05, disc_hi = 0.15, qty_max = 1000000.0;

        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        if (blocks == 0) blocks = 1;
        kernel<<<blocks, threads_per_block>>>(d_shipdate, d_discount, d_quantity, d_price,
                                               date_lo, date_hi, disc_lo, disc_hi, qty_max, d_result);

        cudaDeviceSynchronize();
        cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        printf("Test n=%ld: result=%f\n", n, h_result[0]);

        cudaFree(d_shipdate);
        cudaFree(d_discount);
        cudaFree(d_quantity);
        cudaFree(d_price);
        cudaFree(d_result);

        delete[] h_shipdate;
        delete[] h_discount;
        delete[] h_quantity;
        delete[] h_price;
        delete[] h_result;
    }

    printf("All tests passed!\n");
    return 0;
}
