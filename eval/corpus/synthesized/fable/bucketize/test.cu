// Test the bucketize kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o bucketize_test kernel.cu test.cu
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* values,
                                  const double* bounds,
                                  int* out,
                                  long long n,
                                  long long b);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err));                              \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static bool check(long long n, long long b, std::mt19937_64& rng) {
    // Integer-valued bounds with duplicates, and values that are a mix of
    // exact bounds, values between bounds, and values outside the bounds.
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> bounds(b);
    for (long long j = 0; j < b; ++j) bounds[j] = std::round(normal(rng) * 30.0);
    std::sort(bounds.begin(), bounds.end());
    std::vector<double> values(n);
    for (long long i = 0; i < n; ++i) {
        double v = normal(rng) * 40.0;
        values[i] = (rng() % 10 < 3) ? std::round(v) : v;
    }
    std::vector<int> expected(n);
    for (long long i = 0; i < n; ++i) {
        expected[i] = (int)(std::upper_bound(bounds.begin(), bounds.end(), values[i]) - bounds.begin());
    }

    double *d_values, *d_bounds;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bounds, (b + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds, bounds.data(), b * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(int)));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_values, d_bounds, d_out, n, b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = n == 0 || std::memcmp(out.data(), expected.data(), n * sizeof(int)) == 0;

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("n=%lld b=%lld: %s\n", n, b, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937_64 rng(10);
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long bound_counts[] = {0, 1, 7, 1000};
    int failures = 0;
    for (long long n : sizes) {
        for (long long b : bound_counts) {
            if (!check(n, b, rng)) ++failures;
        }
    }
    return failures ? 1 : 0;
}
