// Test the CUDA coalesce kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void check(int64_t n, std::mt19937_64& rng)
{
    std::normal_distribution<double> normal;
    std::vector<double> a(n), b(n);
    for (double& x : a) x = normal(rng);
    for (double& x : b) x = normal(rng);
    std::vector<uint8_t> valid((n + 7) / 8);
    for (uint8_t& x : valid) x = static_cast<uint8_t>(rng());

    std::vector<double> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        expected[i] = ((valid[i / 8] >> (i % 8)) & 1) ? a[i] : b[i];
    }

    double* d_a = to_device(a);
    uint8_t* d_valid = to_device(valid);
    double* d_b = to_device(b);
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, n * sizeof(double)));

    const int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_a, d_valid, d_b, d_out, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<double> got = to_host(d_out, n);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "coalesce: mismatch at n=%lld\n", static_cast<long long>(n));
        std::exit(1);
    }
    std::printf("coalesce n=%lld passed\n", static_cast<long long>(n));
}

int main()
{
    std::mt19937_64 rng(1);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        check(n, rng);
    }
    return 0;
}
