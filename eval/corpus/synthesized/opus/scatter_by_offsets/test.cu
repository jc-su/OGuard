// GPU test of kernel.cu against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_scatter_by_offsets

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                                  \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(status));                                 \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

constexpr long long kGuard = 64;
constexpr int kUnset = -7;

static void check(bool ok, const char* what, long long n)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL n=%lld: %s\n", n, what);
        std::exit(1);
    }
}

static int default_grid(long long n, int block)
{
    int device = 0;
    int sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const long long blocks = (n + block - 1) / block;
    return static_cast<int>(std::max(1LL, std::min(blocks, 8LL * sms)));
}

static void scatter_and_check(const std::vector<signed char>& flags, int grid, int block)
{
    const long long n = static_cast<long long>(flags.size());
    std::vector<int> offsets(n);
    std::vector<int> want;
    long long total = 0;
    for (long long i = 0; i < n; ++i) {
        offsets[i] = static_cast<int>(total);
        total += flags[i];
        if (flags[i] != 0) {
            want.push_back(static_cast<int>(i));
        }
    }

    signed char* d_flags = nullptr;
    int* d_offsets = nullptr;
    int* d_out = nullptr;
    const std::vector<int> unset(total + kGuard, kUnset);
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_offsets, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, unset.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_flags, flags.data(), n * sizeof(signed char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, unset.data(), unset.size() * sizeof(int), cudaMemcpyHostToDevice));

    kernel<<<grid, block>>>(d_flags, d_offsets, d_out, n, total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(unset.size());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_out));

    check(std::equal(want.begin(), want.end(), out.begin()), "scattered row ids differ", n);
    check(std::all_of(out.begin() + total, out.end(), [](int v) { return v == kUnset; }),
          "write past total", n);
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const double densities[] = {0.0, 0.001, 0.5, 1.0};
    const int block = 256;
    std::mt19937_64 rng(8);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (const long long n : sizes) {
        for (const double density : densities) {
            std::vector<signed char> flags(n);
            for (long long i = 0; i < n; ++i) {
                flags[i] = uniform(rng) < density ? 1 : 0;
            }
            scatter_and_check(flags, default_grid(n, block), block);
            scatter_and_check(flags, 3, 100);
        }
        std::printf("scatter_by_offsets n=%lld: ok\n", n);
    }
    std::printf("PASS\n");
    return 0;
}
