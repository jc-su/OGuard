// Test the CUDA block_sort kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Each size runs with keys over the whole int32 range and with keys from a
// small range plus INT32_MIN and INT32_MAX, which gives many equal keys and
// keys equal to the padding of a short last tile. out starts as the complement
// of the expected keys, so the test also fails when the kernel skips a row.
// kTile and kThreads come from kernel.cu.
#include <algorithm>
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

static void check(int64_t n, bool narrow, std::mt19937_64& rng)
{
    const int32_t choices[] = {INT32_MIN, -2, -1, 0, 1, 2, INT32_MAX};
    std::vector<int32_t> keys(n);
    for (int32_t& x : keys) {
        x = narrow ? choices[rng() % 7] : static_cast<int32_t>(static_cast<uint32_t>(rng()));
    }

    std::vector<int32_t> expected = keys;
    for (int64_t start = 0; start < n; start += kTile) {
        std::sort(expected.begin() + start, expected.begin() + std::min<int64_t>(start + kTile, n));
    }
    std::vector<int32_t> before(n);
    for (int64_t i = 0; i < n; ++i) before[i] = ~expected[i];

    int32_t* d_keys = to_device(keys);
    int32_t* d_out = to_device(before);

    const int64_t tiles = (n + kTile - 1) / kTile;
    if (tiles > 0) {
        kernel<<<static_cast<unsigned>(tiles), kThreads>>>(d_keys, d_out, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<int32_t> got = to_host(d_out, n);

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "block_sort: mismatch at n=%lld narrow=%d\n",
                     static_cast<long long>(n), narrow);
        std::exit(1);
    }
    std::printf("block_sort n=%lld narrow=%d passed\n", static_cast<long long>(n), narrow);
}

int main()
{
    std::mt19937_64 rng(9);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const bool narrow : {false, true}) {
            check(n, narrow, rng);
        }
    }
    return 0;
}
