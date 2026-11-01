// Test the filter_select CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernel.cu"

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define REQUIRE(cond, ...)                                                   \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);        \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// One full wave of resident blocks, or fewer when n needs fewer tiles.
static int grid_for(long long tiles)
{
    int device = 0;
    int sms = 0;
    int threads = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const long long resident = static_cast<long long>(sms) * (threads / BLOCK);
    return static_cast<int>(std::min(tiles, resident));
}

static void run_filter_select(const double* quantity, double threshold, int32_t* sel,
                              int32_t* count, long long n)
{
    CHECK_CUDA(cudaMemset(count, 0, sizeof(int32_t)));
    if (n == 0) {
        return;
    }
    kernel<<<grid_for((n + TILE - 1) / TILE), BLOCK>>>(quantity, threshold, sel, count, n);
    CHECK_CUDA(cudaGetLastError());
}

static void check(long long n, double threshold, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> value(0.0, 50.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<double> quantity(n);
    for (auto& q : quantity) {
        // Some rows are equal to the threshold.
        q = unit(rng) < 0.1 ? threshold : value(rng);
    }

    std::vector<int32_t> expected;
    for (long long i = 0; i < n; ++i) {
        if (quantity[i] < threshold) {
            expected.push_back(static_cast<int32_t>(i));
        }
    }

    double* d_quantity = nullptr;
    int32_t* d_sel = nullptr;
    int32_t* d_count = nullptr;
    CHECK_CUDA(cudaMalloc(&d_quantity, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sel, n * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_quantity, quantity.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sel, 0xff, n * sizeof(int32_t)));  // Every slot is -1.
    const int32_t garbage = 12345;
    CHECK_CUDA(cudaMemcpy(d_count, &garbage, sizeof(int32_t), cudaMemcpyHostToDevice));

    run_filter_select(d_quantity, threshold, d_sel, d_count, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t count = 0;
    std::vector<int32_t> sel(n);
    CHECK_CUDA(cudaMemcpy(&count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sel.data(), d_sel, n * sizeof(int32_t), cudaMemcpyDeviceToHost));

    REQUIRE(count == static_cast<int32_t>(expected.size()), "n=%lld threshold=%g: count %d != %zu",
            n, threshold, count, expected.size());
    std::sort(sel.begin(), sel.begin() + count);
    for (int32_t j = 0; j < count; ++j) {
        REQUIRE(sel[j] == expected[j], "n=%lld threshold=%g: sel mismatch at %d", n, threshold, j);
    }
    for (long long j = count; j < n; ++j) {
        REQUIRE(sel[j] == -1, "n=%lld threshold=%g: write past count at %lld", n, threshold, j);
    }

    CHECK_CUDA(cudaFree(d_quantity));
    CHECK_CUDA(cudaFree(d_sel));
    CHECK_CUDA(cudaFree(d_count));
}

int main()
{
    std::mt19937_64 rng(1);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (double threshold : {-1.0, 24.0, 51.0}) {
            check(n, threshold, rng);
        }
        std::printf("filter_select cuda: n=%lld pass\n", n);
    }
    return 0;
}
