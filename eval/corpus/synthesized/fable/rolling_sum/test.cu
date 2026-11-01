// Test of the rolling_sum CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* values, double* out, long long n, long long window);

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

static bool run_case(long long n, long long window, const std::vector<double>& values, const double* d_values)
{
    // Reference: prefix sums in long double, then differences.
    std::vector<long double> cs(std::max<long long>(n, 1));
    long double acc = 0;
    for (long long i = 0; i < n; ++i) { acc += values[i]; cs[i] = acc; }
    std::vector<double> expected(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i) expected[i] = (double)(i >= window ? cs[i] - cs[i - window] : cs[i]);

    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, std::max<long long>(n, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0xff, std::max<long long>(n, 1) * sizeof(double)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_values, d_out, n, window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        const double diff = fabs(out[i] - expected[i]);
        if (!(diff <= 1e-9 + 1e-12 * fabs(expected[i]))) {
            printf("FAIL n=%lld window=%lld: out[%lld] = %.17g, expected %.17g\n", n, window, i, out[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long windows[] = {1, 2, 7, 256, 1000, 1024};
    bool ok = true;
    for (long long n : sizes) {
        std::vector<double> values(std::max<long long>(n, 1));
        for (long long i = 0; i < n; ++i) values[i] = (double)next_u32() / 2147483648.0 - 1.0;  // [-1, 1)
        double* d_values = nullptr;
        CUDA_CHECK(cudaMalloc(&d_values, std::max<long long>(n, 1) * sizeof(double)));
        if (n > 0) CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        for (long long w : windows) ok = run_case(n, w, values, d_values) && ok;
        CUDA_CHECK(cudaFree(d_values));
        printf("rolling_sum cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
