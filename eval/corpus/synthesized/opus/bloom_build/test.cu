// Test the CUDA bloom_build kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Each size runs with words in {1, 1024, 2^26}, the smallest, a middle, and
// the largest filter. Keys cover the whole int64 range, and a quarter of them
// repeat earlier keys.
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

static std::vector<int64_t> make_keys(int64_t n, std::mt19937_64& rng)
{
    std::vector<int64_t> keys(n);
    for (int64_t i = 0; i < n; ++i) {
        keys[i] = (i > 0 && rng() % 4 == 0) ? keys[rng() % i] : static_cast<int64_t>(rng());
    }
    return keys;
}

static void check(int64_t n, int64_t words, std::mt19937_64& rng)
{
    const std::vector<int64_t> keys = make_keys(n, rng);

    std::vector<uint32_t> expected(words, 0);
    const uint64_t mask = static_cast<uint64_t>(words) * 32 - 1;
    for (const int64_t key : keys) {
        const uint64_t unsigned_key = static_cast<uint64_t>(key);
        for (const uint64_t multiplier : {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL}) {
            const uint64_t p = ((unsigned_key * multiplier) >> 32) & mask;
            expected[p / 32] |= 1u << (p % 32);
        }
    }

    int64_t* d_keys = to_device(keys);
    uint32_t* d_bits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_bits, 0, words * sizeof(uint32_t)));

    const int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_keys, d_bits, n, words);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint32_t> got = to_host(d_bits, words);

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_bits));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "bloom_build: mismatch at n=%lld words=%lld\n",
                     static_cast<long long>(n), static_cast<long long>(words));
        std::exit(1);
    }
    std::printf("bloom_build n=%lld words=%lld passed\n", static_cast<long long>(n),
                static_cast<long long>(words));
}

int main()
{
    std::mt19937_64 rng(6);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const int64_t words : {1LL, 1024LL, 1LL << 26}) {
            check(n, words, rng);
        }
    }
    return 0;
}
