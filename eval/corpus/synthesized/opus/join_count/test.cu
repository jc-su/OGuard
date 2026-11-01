// Test the join_count CUDA kernel on the GPU against a CPU reference.
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

void check(const std::vector<int64_t>& build, const std::vector<int64_t>& probe) {
    const int64_t b = static_cast<int64_t>(build.size());
    const int64_t p = static_cast<int64_t>(probe.size());
    std::vector<int32_t> expected(p);
    for (int64_t i = 0; i < p; ++i) {
        const auto range = std::equal_range(build.begin(), build.end(), probe[i]);
        expected[i] = static_cast<int32_t>(range.second - range.first);
    }

    int64_t* d_build = to_device(build);
    int64_t* d_probe = to_device(probe);
    int32_t* d_counts = to_device(std::vector<int32_t>(p, -1));
    if (p > 0) {
        const int64_t blocks = (p + kThreads - 1) / kThreads;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_build, d_probe, d_counts, b, p);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_counts, p) == expected);

    CHECK_CUDA(cudaFree(d_build));
    CHECK_CUDA(cudaFree(d_probe));
    CHECK_CUDA(cudaFree(d_counts));
}

int main() {
    std::mt19937_64 rng(3);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t b : sizes) {
        for (int64_t p : sizes) {
            // Keys drawn from a range near b give many duplicates and misses.
            const int64_t span = std::max<int64_t>(b / 4, 1);
            std::vector<int64_t> build(b);
            std::vector<int64_t> probe(p);
            for (int64_t& x : build) x = static_cast<int64_t>(rng() % (2 * span)) - span;
            for (int64_t& x : probe) x = static_cast<int64_t>(rng() % (4 * span)) - 2 * span;
            std::sort(build.begin(), build.end());
            check(build, probe);
        }
    }
    const int64_t lo = std::numeric_limits<int64_t>::min();
    const int64_t hi = std::numeric_limits<int64_t>::max();
    check({lo, lo, -1, 0, 0, 0, 7, hi, hi, hi}, {lo, -1, 0, 1, 7, hi, hi - 1});
    check(std::vector<int64_t>(1000003, 5), {4, 5, 6});
    std::printf("join_count cuda: all tests passed\n");
    return 0;
}
