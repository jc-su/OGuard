// Test the transpose CUDA kernel on the GPU against a CPU reference.
// The shapes cover rows * cols in {0, 1, 1000, 1000003} and shapes that are not
// multiples of the 32 x 32 tile.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

void check(int64_t rows, int64_t cols, std::mt19937_64& rng) {
    const int64_t size = rows * cols;
    std::vector<int32_t> source(size);
    for (int32_t& x : source) x = static_cast<int32_t>(rng());
    std::vector<int32_t> expected(size);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) expected[j * rows + i] = source[i * cols + j];
    }

    int32_t* d_source = to_device(source);
    int32_t* d_out = to_device(std::vector<int32_t>(size, -1));
    const int64_t ntiles = ((rows + kTile - 1) / kTile) * ((cols + kTile - 1) / kTile);
    if (ntiles > 0) {
        kernel<<<static_cast<unsigned>(ntiles), kThreads>>>(d_source, d_out, rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    REQUIRE(to_host(d_out, size) == expected);

    CHECK_CUDA(cudaFree(d_source));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    std::mt19937_64 rng(6);
    const std::pair<int64_t, int64_t> shapes[] = {
        {0, 0},       {0, 5},       {5, 0},   {1, 1},     {1, 1000},
        {1000, 1},    {25, 40},     {40, 25}, {1, 1000003}, {1000003, 1},
        {33, 31},     {999, 1001},  {1024, 1024},
    };
    for (const auto& [rows, cols] : shapes) check(rows, cols, rng);
    std::printf("transpose cuda: all tests passed\n");
    return 0;
}
