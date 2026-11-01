// Test the gather CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
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

// One full wave of resident blocks, or fewer when m needs fewer blocks.
static int grid_for(long long blocks)
{
    int device = 0;
    int sms = 0;
    int threads = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const long long resident = static_cast<long long>(sms) * (threads / BLOCK);
    return static_cast<int>(std::min(blocks, resident));
}

static void run_gather(const double* col, const int32_t* sel, double* out, long long m, long long n)
{
    if (m == 0) {
        return;
    }
    kernel<<<grid_for((m + BLOCK - 1) / BLOCK), BLOCK>>>(col, sel, out, m, n);
    CHECK_CUDA(cudaGetLastError());
}

static void check(long long n, long long m, std::mt19937_64& rng)
{
    std::normal_distribution<double> value(0.0, 1.0);
    std::vector<double> col(n);
    for (auto& c : col) {
        c = value(rng);
    }
    std::vector<int32_t> sel(m);
    if (m > 0) {
        std::uniform_int_distribution<int32_t> row(0, static_cast<int32_t>(n - 1));
        for (auto& s : sel) {
            s = row(rng);
        }
    }

    double* d_col = nullptr;
    int32_t* d_sel = nullptr;
    double* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_col, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sel, m * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, m * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_col, col.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sel, sel.data(), m * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0xff, m * sizeof(double)));  // Every value is NaN.

    run_gather(d_col, d_sel, d_out, m, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> out(m);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, m * sizeof(double), cudaMemcpyDeviceToHost));
    for (long long i = 0; i < m; ++i) {
        REQUIRE(out[i] == col[sel[i]], "n=%lld m=%lld: out mismatch at %lld", n, m, i);
    }

    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_sel));
    CHECK_CUDA(cudaFree(d_out));
}

int main()
{
    std::mt19937_64 rng(2);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        // sel can hold row ids only when col has rows.
        const std::set<long long> ms = n > 0 ? std::set<long long>{n, n / 2, 2 * n}
                                             : std::set<long long>{0};
        for (long long m : ms) {
            check(n, m, rng);
        }
        std::printf("gather cuda: n=%lld pass\n", n);
    }
    return 0;
}
