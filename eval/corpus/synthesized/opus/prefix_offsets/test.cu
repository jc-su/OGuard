// Test the prefix_offsets CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
constexpr int32_t kMaxLength = 1 << 20;

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

void check(const std::vector<int32_t>& lengths) {
    const int64_t n = static_cast<int64_t>(lengths.size());
    std::vector<int32_t> expected(n + 1);
    int64_t sum = 0;
    expected[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += lengths[i];
        expected[i + 1] = static_cast<int32_t>(sum);
    }

    int32_t* d_lengths = to_device(lengths);
    int32_t* d_offsets = to_device(std::vector<int32_t>(n + 1, -7));
    kernel<<<1, kThreads>>>(d_lengths, d_offsets, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_offsets, n + 1) == expected);

    CHECK_CUDA(cudaFree(d_lengths));
    CHECK_CUDA(cudaFree(d_offsets));
}

int main() {
    std::mt19937_64 rng(5);
    for (int64_t n : {0LL, 1LL, 1000LL, 4095LL, 4096LL, 4097LL, 1000003LL}) {
        // The largest length that keeps offsets[n] within int32.
        const int64_t top = std::min<int64_t>(kMaxLength, INT32_MAX / std::max<int64_t>(n, 1));
        std::vector<int32_t> lengths(n);
        for (int32_t& x : lengths) x = static_cast<int32_t>(rng() % (top + 1));
        check(lengths);
        check(std::vector<int32_t>(n, 0));
    }
    check(std::vector<int32_t>(2047, kMaxLength));
    std::printf("prefix_offsets cuda: all tests passed\n");
    return 0;
}
