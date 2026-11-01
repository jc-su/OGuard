// Test the projection CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test

#include <algorithm>
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

// One full wave of resident blocks, or fewer when n needs fewer blocks.
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

static void run_projection(const double* price, const double* discount, double* out, long long n)
{
    if (n == 0) {
        return;
    }
    kernel<<<grid_for((n + BLOCK - 1) / BLOCK), BLOCK>>>(price, discount, out, n);
    CHECK_CUDA(cudaGetLastError());
}

static void check(long long n, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> price_value(900.0, 105000.0);
    std::uniform_int_distribution<int> discount_cents(0, 10);
    std::vector<double> price(n);
    std::vector<double> discount(n);
    for (long long i = 0; i < n; ++i) {
        price[i] = price_value(rng);
        discount[i] = discount_cents(rng) / 100.0;
    }

    double* d_price = nullptr;
    double* d_discount = nullptr;
    double* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_price, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_discount, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_price, price.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_discount, discount.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(double)));  // Every value is NaN.

    run_projection(d_price, d_discount, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> out(n);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    // The kernel does one subtraction and one multiplication in double, the
    // same operations as the reference, so the results are equal bit for bit.
    for (long long i = 0; i < n; ++i) {
        const double expected = price[i] * (1.0 - discount[i]);
        REQUIRE(out[i] == expected, "n=%lld: out mismatch at %lld", n, i);
    }

    CHECK_CUDA(cudaFree(d_price));
    CHECK_CUDA(cudaFree(d_discount));
    CHECK_CUDA(cudaFree(d_out));
}

int main()
{
    std::mt19937_64 rng(3);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        check(n, rng);
        std::printf("projection cuda: n=%lld pass\n", n);
    }
    return 0;
}
