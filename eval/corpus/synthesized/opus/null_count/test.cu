// Test the null_count CUDA kernel on the GPU against a CPU reference.
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
constexpr int kBlocksPerSm = 4;
constexpr int64_t kInitial = 12345;

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

void check(int64_t n, double null_density, int sms, std::mt19937_64& rng) {
    const int64_t nbytes = (n + 7) / 8;
    // Bits past row n - 1 in the last byte start random; they must not count.
    std::vector<uint8_t> valid(nbytes);
    for (uint8_t& byte : valid) byte = static_cast<uint8_t>(rng());
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int64_t expected = kInitial;
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t bit = static_cast<uint8_t>(1u << (i % 8));
        if (unit(rng) < null_density) {
            valid[i / 8] &= static_cast<uint8_t>(~bit);
            ++expected;
        } else {
            valid[i / 8] |= bit;
        }
    }

    uint8_t* d_valid = to_device(valid);
    int64_t* d_count = to_device(std::vector<int64_t>(1, kInitial));
    if (nbytes > 0) {
        const int64_t blocks =
            std::min<int64_t>((nbytes + kThreads - 1) / kThreads, int64_t{sms} * kBlocksPerSm);
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_valid, d_count, n);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_count, 1)[0] == expected);

    CHECK_CUDA(cudaFree(d_valid));
    CHECK_CUDA(cudaFree(d_count));
}

int main() {
    int sms = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937_64 rng(4);
    for (int64_t n : {0LL, 1LL, 7LL, 8LL, 9LL, 1000LL, 1000003LL}) {
        for (double null_density : {0.0, 0.1, 0.5, 1.0}) {
            check(n, null_density, sms, rng);
        }
    }
    std::printf("null_count cuda: all tests passed\n");
    return 0;
}
