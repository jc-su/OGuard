// Test the hash_partition CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(err_));                                   \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define REQUIRE(cond)                                                                 \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,    \
                         #cond);                                                      \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

constexpr int kThreads = 256;
constexpr int kBlocksPerSm = 4;

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty()) {
        CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    return device;
}

template <typename T>
std::vector<T> to_host(const T* device, size_t size) {
    std::vector<T> host(size);
    if (size != 0) {
        CHECK_CUDA(cudaMemcpy(host.data(), device, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host;
}

void check(const std::vector<int64_t>& keys, int64_t p, int sms) {
    const int64_t n = static_cast<int64_t>(keys.size());
    std::vector<int32_t> expected_part(n);
    std::vector<int32_t> expected_counts(p, 0);
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t h = (static_cast<uint64_t>(keys[i]) * 0x9E3779B97F4A7C15ull) >> 32;
        expected_part[i] = static_cast<int32_t>(h % static_cast<uint64_t>(p));
        ++expected_counts[expected_part[i]];
    }

    int64_t* d_keys = to_device(keys);
    int32_t* d_part = to_device(std::vector<int32_t>(n, -1));
    int32_t* d_counts = to_device(std::vector<int32_t>(p, 0));
    if (n > 0) {
        const int64_t blocks =
            std::min<int64_t>((n + kThreads - 1) / kThreads, int64_t{sms} * kBlocksPerSm);
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_keys, d_part, d_counts, n, p);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_part, n) == expected_part);
    REQUIRE(to_host(d_counts, p) == expected_counts);

    CHECK_CUDA(cudaFree(d_keys));
    CHECK_CUDA(cudaFree(d_part));
    CHECK_CUDA(cudaFree(d_counts));
}

int main() {
    int sms = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937_64 rng(7);
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (int64_t p : {1LL, 7LL, 1000LL, 1024LL}) {
            std::vector<int64_t> keys(n);
            for (int64_t& x : keys) x = static_cast<int64_t>(rng());
            check(keys, p, sms);
        }
    }
    std::vector<int64_t> sequence(1000);
    for (int64_t i = 0; i < 1000; ++i) sequence[i] = i - 500;
    check(sequence, 16, sms);
    check(std::vector<int64_t>(1000003, 42), 1024, sms);
    check({std::numeric_limits<int64_t>::min(), -1, 0, 1, std::numeric_limits<int64_t>::max()},
          3, sms);
    std::printf("hash_partition cuda: all tests passed\n");
    return 0;
}
