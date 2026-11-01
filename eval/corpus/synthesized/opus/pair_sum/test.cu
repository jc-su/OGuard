// Test the pair_sum CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
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

void check(int64_t m, int64_t l, int64_t r, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1e6);
    std::vector<int32_t> left_idx(m);
    std::vector<int32_t> right_idx(m);
    std::vector<double> left(l);
    std::vector<double> right(r);
    for (int32_t& x : left_idx) x = static_cast<int32_t>(rng() % l);
    for (int32_t& x : right_idx) x = static_cast<int32_t>(rng() % r);
    for (double& x : left) x = normal(rng);
    for (double& x : right) x = normal(rng);
    std::vector<double> expected(m);
    for (int64_t i = 0; i < m; ++i) expected[i] = left[left_idx[i]] + right[right_idx[i]];

    int32_t* d_left_idx = to_device(left_idx);
    int32_t* d_right_idx = to_device(right_idx);
    double* d_left = to_device(left);
    double* d_right = to_device(right);
    double* d_out = to_device(std::vector<double>(m, NAN));
    if (m > 0) {
        const int64_t blocks = (m + kThreads - 1) / kThreads;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_left_idx, d_right_idx, d_left,
                                                             d_right, d_out, m, l, r);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    // Both sides compute one IEEE double addition, so the bits must match.
    const std::vector<double> out = to_host(d_out, m);
    REQUIRE(m == 0 || std::memcmp(out.data(), expected.data(), m * sizeof(double)) == 0);

    CHECK_CUDA(cudaFree(d_left_idx));
    CHECK_CUDA(cudaFree(d_right_idx));
    CHECK_CUDA(cudaFree(d_left));
    CHECK_CUDA(cudaFree(d_right));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    std::mt19937_64 rng(10);
    // (l, r) pairs: a small table and a large table on either side.
    const std::pair<int64_t, int64_t> tables[] = {{1, 1}, {777, 1000003}, {1000003, 13}};
    for (int64_t m : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const auto& [l, r] : tables) check(m, l, r, rng);
    }
    check(0, 0, 0, rng);
    std::printf("pair_sum cuda: all tests passed\n");
    return 0;
}
