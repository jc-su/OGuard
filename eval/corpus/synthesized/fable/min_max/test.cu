// Test of the min_max CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* values, double* result, long long n);

#define CUDA_CHECK(x)                                                                        \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t next_u32()
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static bool run_case(long long n, double shift)
{
    std::vector<double> values(std::max<long long>(n, 1));
    double expected_min = INFINITY, expected_max = -INFINITY;
    for (long long i = 0; i < n; ++i) {
        values[i] = (double)(int32_t)next_u32() * 0.5 + shift;  // about [-1e9, 1e9] + shift
        expected_min = std::min(expected_min, values[i]);
        expected_max = std::max(expected_max, values[i]);
    }

    double* d_values = nullptr;
    double* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_values, std::max<long long>(n, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, 2 * sizeof(double)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    const double init[2] = {INFINITY, -INFINITY};
    CUDA_CHECK(cudaMemcpy(d_result, init, 2 * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned blocks = (unsigned)std::min<long long>(2048, std::max<long long>(1, (n + 255) / 256));
    kernel<<<blocks, 256>>>(d_values, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double result[2];
    CUDA_CHECK(cudaMemcpy(result, d_result, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_result));

    if (result[0] != expected_min || result[1] != expected_max) {
        printf("FAIL n=%lld shift=%g: got [%g, %g], expected [%g, %g]\n", n, shift, result[0], result[1], expected_min, expected_max);
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const double shifts[] = {0.0, 2e9, -2e9};  // mixed signs, all positive, all negative
    bool ok = true;
    for (long long n : sizes) {
        for (double s : shifts) ok = run_case(n, s) && ok;
        printf("min_max cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
