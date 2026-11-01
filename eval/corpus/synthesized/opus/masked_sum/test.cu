#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(err_));                                   \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

static const int THREADS = 256;
static const int BLOCKS_PER_SM = 8;

static bool run_case(int64_t n, int sms, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> value_dist(-100.0, 100.0);
    std::uniform_int_distribution<int> mask_dist(-2, 2);
    std::vector<double> values(n);
    std::vector<int8_t> mask(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = value_dist(rng);
        mask[i] = static_cast<int8_t>(mask_dist(rng) * 50);
    }
    long double expected = 0.0L;
    long double magnitude = 0.0L;
    for (int64_t i = 0; i < n; ++i) {
        if (mask[i] != 0) {
            expected += values[i];
            magnitude += std::fabs(values[i]);
        }
    }

    double* d_values = nullptr;
    int8_t* d_mask = nullptr;
    double* d_result = nullptr;
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_mask, n + 1));
    CHECK(cudaMalloc(&d_result, sizeof(double)));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, mask.data(), n, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_result, 0, sizeof(double)));
    if (n > 0) {
        const int64_t needed = (n + THREADS - 1) / THREADS;
        const unsigned blocks =
            static_cast<unsigned>(std::min<int64_t>(needed, int64_t(sms) * BLOCKS_PER_SM));
        kernel<<<blocks, THREADS>>>(d_values, d_mask, d_result, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    double got = 0.0;
    CHECK(cudaMemcpy(&got, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_mask));
    CHECK(cudaFree(d_result));

    const double bound = 1e-12 * (static_cast<double>(magnitude) + 1.0);
    if (std::fabs(got - static_cast<double>(expected)) > bound) {
        std::fprintf(stderr, "n=%lld: result = %.17g, expected %.17g\n", (long long)n, got,
                     static_cast<double>(expected));
        return false;
    }
    return true;
}

int main()
{
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937_64 rng(6);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t n : sizes) {
        ok = run_case(n, sms, rng) && ok;
    }
    std::printf("masked_sum cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
