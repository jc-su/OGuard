// Test the string_prefix CUDA kernel on the GPU against a CPU reference.
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

void check(int64_t n, int64_t k, int32_t first, std::mt19937_64& rng) {
    std::vector<int32_t> offsets(n + 1);
    offsets[0] = first;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t length = rng() % 10 == 0 ? 0 : static_cast<int32_t>(rng() % 100);
        offsets[i + 1] = offsets[i] + length;
    }
    std::vector<uint8_t> chars(offsets[n]);
    for (uint8_t& c : chars) c = static_cast<uint8_t>(rng());

    std::vector<uint8_t> expected(n * k, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t length = std::min<int64_t>(k, offsets[i + 1] - offsets[i]);
        for (int64_t c = 0; c < length; ++c) expected[i * k + c] = chars[offsets[i] + c];
    }

    int32_t* d_offsets = to_device(offsets);
    uint8_t* d_chars = to_device(chars);
    uint8_t* d_out = to_device(std::vector<uint8_t>(n * k, 0xA5));
    const int64_t total = n * k;
    if (total > 0) {
        const int64_t blocks = (total + kThreads - 1) / kThreads;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_offsets, d_chars, d_out, n, k);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_out, n * k) == expected);

    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_chars));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    std::mt19937_64 rng(2);
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (int64_t k : {1LL, 5LL, 64LL}) {
            for (int32_t first : {0, 3}) {
                check(n, k, first, rng);
            }
        }
    }
    std::printf("string_prefix cuda: all tests passed\n");
    return 0;
}
