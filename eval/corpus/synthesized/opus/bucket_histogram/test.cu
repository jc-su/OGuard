#include <algorithm>
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

static bool run_case(int64_t n, int64_t b, int sms, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> bound_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> value_dist(-12.0, 12.0);
    std::vector<double> bounds(b);
    for (int64_t k = 0; k < b; ++k) {
        bounds[k] = bound_dist(rng);
    }
    std::sort(bounds.begin(), bounds.end());
    if (b > 3) {
        bounds[2] = bounds[1];
    }
    std::vector<double> values(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = (b > 0 && rng() % 4 == 0) ? bounds[rng() % b] : value_dist(rng);
    }
    std::vector<int32_t> expected(b + 1, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t c = std::upper_bound(bounds.begin(), bounds.end(), values[i]) - bounds.begin();
        ++expected[c];
    }

    double* d_values = nullptr;
    double* d_bounds = nullptr;
    int32_t* d_counts = nullptr;
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_bounds, (b + 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_counts, (b + 1) * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bounds, bounds.data(), b * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_counts, 0, (b + 1) * sizeof(int32_t)));
    if (n > 0) {
        const int64_t needed = (n + THREADS - 1) / THREADS;
        const unsigned blocks =
            static_cast<unsigned>(std::min<int64_t>(needed, int64_t(sms) * BLOCKS_PER_SM));
        kernel<<<blocks, THREADS>>>(d_values, d_bounds, d_counts, n, b);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got(b + 1);
    CHECK(cudaMemcpy(got.data(), d_counts, (b + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_bounds));
    CHECK(cudaFree(d_counts));

    for (int64_t k = 0; k <= b; ++k) {
        if (got[k] != expected[k]) {
            std::fprintf(stderr, "n=%lld b=%lld: counts[%lld] = %d, expected %d\n",
                         (long long)n, (long long)b, (long long)k, got[k], expected[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937_64 rng(7);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t bound_counts[] = {0, 1, 15, 4095, 10000};
    bool ok = true;
    for (int64_t n : sizes) {
        for (int64_t b : bound_counts) {
            ok = run_case(n, b, sms, rng) && ok;
        }
    }
    std::printf("bucket_histogram cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
