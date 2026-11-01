// Test the CUDA lag kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Each size runs with k in {1, 7, 1024}. A row starts a new partition with
// probability 1 / (2k), so partitions average 2k rows and both outcomes of the
// test i - k >= first[i] occur.
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

static void check(int64_t n, int64_t k, std::mt19937_64& rng)
{
    std::normal_distribution<double> normal;
    std::bernoulli_distribution starts(1.0 / (2.0 * static_cast<double>(k)));
    std::vector<double> values(n);
    std::vector<int32_t> first(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = normal(rng);
        first[i] = (i == 0 || starts(rng)) ? static_cast<int32_t>(i) : first[i - 1];
    }

    std::vector<double> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        expected[i] = i - k >= first[i] ? values[i - k] : std::numeric_limits<double>::quiet_NaN();
    }

    double* d_values = to_device(values);
    int32_t* d_first = to_device(first);
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, n * sizeof(double)));

    const int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_values, d_first, d_out, n, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<double> got = to_host(d_out, n);

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "lag: mismatch at n=%lld k=%lld\n", static_cast<long long>(n),
                     static_cast<long long>(k));
        std::exit(1);
    }
    std::printf("lag n=%lld k=%lld passed\n", static_cast<long long>(n), static_cast<long long>(k));
}

int main()
{
    std::mt19937_64 rng(5);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const int64_t k : {1LL, 7LL, 1024LL}) {
            check(n, k, rng);
        }
    }
    return 0;
}
