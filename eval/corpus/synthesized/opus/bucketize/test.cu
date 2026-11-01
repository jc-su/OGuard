// Test the CUDA bucketize kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Each size runs with b in {0, 1, 37, 100000}. Bounds and values are rounded
// to one decimal, so bounds repeat and values often equal a bound. The bounds
// start with -inf when b >= 3, and a few values are -inf or +inf.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(err_));                      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kThreads = 256;

template <typename T>
static T* to_device(const std::vector<T>& host)
{
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return ptr;
}

template <typename T>
static std::vector<T> to_host(const T* ptr, size_t count)
{
    std::vector<T> host(count);
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host;
}

template <typename T>
static bool same_bytes(const std::vector<T>& got, const std::vector<T>& expected)
{
    return got.size() == expected.size() &&
           (got.empty() || std::memcmp(got.data(), expected.data(), got.size() * sizeof(T)) == 0);
}

static void check(int64_t n, int64_t b, std::mt19937_64& rng)
{
    const double inf = std::numeric_limits<double>::infinity();
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;
    std::vector<double> bounds(b);
    for (double& x : bounds) x = std::round(normal(rng) * 30.0) / 10.0;
    std::sort(bounds.begin(), bounds.end());
    if (b >= 3) bounds[0] = -inf;
    std::vector<double> values(n);
    for (double& x : values) {
        const double u = uniform(rng);
        x = u < 0.01 ? -inf : u > 0.99 ? inf : std::round(normal(rng) * 35.0) / 10.0;
    }

    std::vector<int32_t> expected(n), before(n);
    for (int64_t i = 0; i < n; ++i) {
        expected[i] = static_cast<int32_t>(std::upper_bound(bounds.begin(), bounds.end(), values[i]) -
                                           bounds.begin());
        before[i] = ~expected[i];
    }

    double* d_values = to_device(values);
    double* d_bounds = to_device(bounds);
    int32_t* d_out = to_device(before);

    const int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_values, d_bounds, d_out, n, b);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<int32_t> got = to_host(d_out, n);

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "bucketize: mismatch at n=%lld b=%lld\n", static_cast<long long>(n),
                     static_cast<long long>(b));
        std::exit(1);
    }
    std::printf("bucketize n=%lld b=%lld passed\n", static_cast<long long>(n),
                static_cast<long long>(b));
}

int main()
{
    std::mt19937_64 rng(10);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const int64_t b : {0LL, 1LL, 37LL, 100000LL}) {
            check(n, b, rng);
        }
    }
    return 0;
}
