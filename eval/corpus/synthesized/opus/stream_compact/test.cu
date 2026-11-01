// Test the stream_compact CUDA kernel on the GPU against a CPU reference.
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
constexpr int64_t kSentinel = 0x5A5A5A5A5A5A5A5ALL;

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

void check(int64_t n, double density, std::mt19937_64& rng) {
    std::vector<int64_t> values(n);
    std::vector<int8_t> keep(n);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = static_cast<int64_t>(rng());
        int8_t flag = static_cast<int8_t>(rng());
        if (flag == 0) flag = 1;
        keep[i] = unit(rng) < density ? flag : 0;
    }
    std::vector<int64_t> expected;
    for (int64_t i = 0; i < n; ++i) {
        if (keep[i] != 0) expected.push_back(values[i]);
    }

    int64_t* d_values = to_device(values);
    int8_t* d_keep = to_device(keep);
    int64_t* d_out = to_device(std::vector<int64_t>(n, kSentinel));
    int32_t* d_count = to_device(std::vector<int32_t>(1, 0));
    if (n > 0) {
        const int64_t blocks = (n + kThreads - 1) / kThreads;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_values, d_keep, d_out, d_count, n);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const int32_t count = to_host(d_count, 1)[0];
    std::vector<int64_t> out = to_host(d_out, n);

    REQUIRE(count == static_cast<int64_t>(expected.size()));
    std::vector<int64_t> got(out.begin(), out.begin() + count);
    std::sort(got.begin(), got.end());
    std::sort(expected.begin(), expected.end());
    REQUIRE(got == expected);
    for (int64_t i = count; i < n; ++i) REQUIRE(out[i] == kSentinel);

    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_keep));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_count));
}

int main() {
    std::mt19937_64 rng(1);
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (double density : {0.0, 0.01, 0.5, 1.0}) {
            check(n, density, rng);
        }
    }
    std::printf("stream_compact cuda: all tests passed\n");
    return 0;
}
