// Test the window_max CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

// Sliding window maximum with a monotonic deque of row numbers.
std::vector<int64_t> reference(const std::vector<int64_t>& values, int64_t w) {
    const int64_t n = static_cast<int64_t>(values.size());
    std::vector<int64_t> result(n);
    std::deque<int64_t> rows;
    for (int64_t i = 0; i < n; ++i) {
        while (!rows.empty() && values[rows.back()] <= values[i]) rows.pop_back();
        rows.push_back(i);
        if (rows.front() <= i - w) rows.pop_front();
        result[i] = values[rows.front()];
    }
    return result;
}

void check(const std::vector<int64_t>& values, int64_t w) {
    const int64_t n = static_cast<int64_t>(values.size());
    const std::vector<int64_t> expected = reference(values, w);

    int64_t* d_values = to_device(values);
    int64_t* d_out = to_device(std::vector<int64_t>(n, 7));
    if (n > 0) {
        const int64_t blocks = (n + kOutTile - 1) / kOutTile;
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_values, d_out, n, w);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_out, n) == expected);

    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    std::mt19937_64 rng(9);
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (int64_t w : {1LL, 2LL, 3LL, 100LL, 512LL, 1000LL, 1024LL}) {
            std::vector<int64_t> values(n);
            for (int64_t& x : values) x = static_cast<int64_t>(rng());
            check(values, w);
        }
    }
    std::vector<int64_t> descending(1000003);
    std::vector<int64_t> ascending(1000003);
    for (int64_t i = 0; i < 1000003; ++i) {
        descending[i] = 1000003 - i;
        ascending[i] = i - 1000003;
    }
    for (int64_t w : {1LL, 777LL, 1024LL}) {
        check(descending, w);
        check(ascending, w);
    }
    check(std::vector<int64_t>(5000, std::numeric_limits<int64_t>::min()), 1024);
    std::printf("window_max cuda: all tests passed\n");
    return 0;
}
