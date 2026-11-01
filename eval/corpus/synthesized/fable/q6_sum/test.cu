// Test for q6_sum: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int* shipdate, const double* discount, const double* quantity,
                                  const double* price, int date_lo, int date_hi, double disc_lo,
                                  double disc_hi, double qty_max, double* result, long long n);

#define CHECK(call)                                                                    \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double next_double() { return (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

static const int THREADS = 256;
static const int MAX_BLOCKS = 2048;
static int grid_for(long long n) {
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return (int)b;
}

static bool run_case(long long n, bool timed) {
    const int date_lo = 2000, date_hi = 2365;
    const double disc_lo = 0.05, disc_hi = 0.07, qty_max = 24.0;
    std::vector<int> shipdate(n);
    std::vector<double> discount(n), quantity(n), price(n);
    double expected = 0.0;
    for (long long i = 0; i < n; ++i) {
        shipdate[i] = (int)(next_u64() % 2557u);
        discount[i] = std::floor(next_double() * 11.0) / 100.0;  // 0.00 .. 0.10 like TPC-H
        quantity[i] = 1.0 + std::floor(next_double() * 50.0);
        price[i] = 1.0 + next_double() * 100000.0;
        if (shipdate[i] >= date_lo && shipdate[i] < date_hi && discount[i] >= disc_lo &&
            discount[i] <= disc_hi && quantity[i] < qty_max)
            expected += price[i] * discount[i];
    }

    int* d_shipdate = nullptr;
    double* d_discount = nullptr;
    double* d_quantity = nullptr;
    double* d_price = nullptr;
    double* d_result = nullptr;
    const long long cap = std::max<long long>(n, 1);
    CHECK(cudaMalloc(&d_shipdate, cap * sizeof(int)));
    CHECK(cudaMalloc(&d_discount, cap * sizeof(double)));
    CHECK(cudaMalloc(&d_quantity, cap * sizeof(double)));
    CHECK(cudaMalloc(&d_price, cap * sizeof(double)));
    CHECK(cudaMalloc(&d_result, sizeof(double)));
    CHECK(cudaMemcpy(d_shipdate, shipdate.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_discount, discount.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_quantity, quantity.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_price, price.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_result, 0, sizeof(double)));

    kernel<<<grid_for(n), THREADS>>>(d_shipdate, d_discount, d_quantity, d_price, date_lo, date_hi,
                                     disc_lo, disc_hi, qty_max, d_result, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double result = 0.0;
    CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    const bool ok = std::fabs(result - expected) <= 1e-9 * std::fabs(expected) + 1e-9;
    printf("  n=%lld: %s (result=%.6f expected=%.6f)\n", n, ok ? "ok" : "MISMATCH", result, expected);

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_result, 0, sizeof(double)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_shipdate, d_discount, d_quantity, d_price, date_lo,
                                             date_hi, disc_lo, disc_hi, qty_max, d_result, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_shipdate));
    CHECK(cudaFree(d_discount));
    CHECK(cudaFree(d_quantity));
    CHECK(cudaFree(d_price));
    CHECK(cudaFree(d_result));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s q6_sum\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
