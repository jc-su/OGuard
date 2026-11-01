// Test the CUDA gather_nullable kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
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

static void check(int64_t m, int64_t n, std::mt19937_64& rng)
{
    std::normal_distribution<double> normal;
    std::vector<double> col(n);
    for (double& x : col) x = normal(rng);
    std::uniform_int_distribution<int32_t> pick(-1, static_cast<int32_t>(n - 1));
    std::vector<int32_t> idx(m);
    for (int32_t& x : idx) x = pick(rng);

    std::vector<double> expected(m);
    for (int64_t i = 0; i < m; ++i) {
        expected[i] = idx[i] >= 0 ? col[idx[i]] : std::numeric_limits<double>::quiet_NaN();
    }

    double* d_col = to_device(col);
    int32_t* d_idx = to_device(idx);
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, m * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, m * sizeof(double)));

    const int64_t blocks = (m + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_col, d_idx, d_out, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<double> got = to_host(d_out, m);

    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "gather_nullable: mismatch at m=%lld n=%lld\n",
                     static_cast<long long>(m), static_cast<long long>(n));
        std::exit(1);
    }
    std::printf("gather_nullable m=%lld n=%lld passed\n",
                static_cast<long long>(m), static_cast<long long>(n));
}

int main()
{
    std::mt19937_64 rng(2);
    for (const int64_t size : {0LL, 1LL, 1000LL, 1000003LL}) {
        check(size, size, rng);
        check(size, size / 2, rng);
    }
    return 0;
}
