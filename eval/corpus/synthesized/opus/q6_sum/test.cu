// Test the q6_sum CUDA kernel on the GPU against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

struct Params {
    int32_t date_lo;
    int32_t date_hi;
    double disc_lo;
    double disc_hi;
    double qty_max;
};

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

// Adds the Q6 revenue of the n rows to result[0], which is zero on entry.
static void run_q6_sum(const int32_t* shipdate, const double* discount, const double* quantity,
                       const double* price, const Params& p, double* result, long long n)
{
    if (n == 0) {
        return;
    }
    kernel<<<grid_for((n + BLOCK - 1) / BLOCK), BLOCK>>>(shipdate, discount, quantity, price,
                                                         p.date_lo, p.date_hi, p.disc_lo,
                                                         p.disc_hi, p.qty_max, result, n);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
static T* to_device(const std::vector<T>& host)
{
    T* device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

static void check(long long n, const Params& p, std::mt19937_64& rng)
{
    // Days of 1992-01-01 to 1998-12-31, as in TPC-H lineitem.
    std::uniform_int_distribution<int32_t> day(8035, 10591);
    std::uniform_int_distribution<int> discount_cents(0, 10);
    std::uniform_int_distribution<int> quantity_value(1, 50);
    std::uniform_real_distribution<double> price_value(900.0, 105000.0);
    std::vector<int32_t> shipdate(n);
    std::vector<double> discount(n);
    std::vector<double> quantity(n);
    std::vector<double> price(n);
    for (long long i = 0; i < n; ++i) {
        shipdate[i] = day(rng);
        discount[i] = discount_cents(rng) / 100.0;
        quantity[i] = quantity_value(rng);
        price[i] = price_value(rng);
    }

    long double expected = 0.0L;
    for (long long i = 0; i < n; ++i) {
        if (p.date_lo <= shipdate[i] && shipdate[i] < p.date_hi && p.disc_lo <= discount[i] &&
            discount[i] <= p.disc_hi && quantity[i] < p.qty_max) {
            expected += static_cast<long double>(price[i] * discount[i]);
        }
    }

    int32_t* d_shipdate = to_device(shipdate);
    double* d_discount = to_device(discount);
    double* d_quantity = to_device(quantity);
    double* d_price = to_device(price);
    double* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(double)));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(double)));

    run_q6_sum(d_shipdate, d_discount, d_quantity, d_price, p, d_result, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    double got = 0.0;
    CHECK_CUDA(cudaMemcpy(&got, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    // The GPU adds the terms in a different order, so allow rounding error.
    const double want = static_cast<double>(expected);
    REQUIRE(std::fabs(got - want) <= 1e-10 * std::fabs(want),
            "n=%lld date=[%d, %d): got %.17g, expected %.17g", n, p.date_lo, p.date_hi, got, want);

    CHECK_CUDA(cudaFree(d_shipdate));
    CHECK_CUDA(cudaFree(d_discount));
    CHECK_CUDA(cudaFree(d_quantity));
    CHECK_CUDA(cudaFree(d_price));
    CHECK_CUDA(cudaFree(d_result));
}

int main()
{
    const double inf = std::numeric_limits<double>::infinity();
    // TPC-H Q6 with dates as days since 1970-01-01: shipdate in
    // [1994-01-01, 1995-01-01), discount in [0.05, 0.07], quantity < 24.
    const Params q6{8766, 9131, 0.05, 0.07, 24.0};
    // Bounds that keep every row.
    const Params all_rows{std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(),
                          -inf, inf, inf};

    std::mt19937_64 rng(4);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const Params& p : {q6, all_rows}) {
            check(n, p, rng);
        }
        std::printf("q6_sum cuda: n=%lld pass\n", n);
    }
    return 0;
}
