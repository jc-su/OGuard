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

static bool run_case(int64_t m, int64_t n, std::mt19937_64& rng)
{
    std::vector<int32_t> idx(m);
    std::vector<double> values(m);
    std::uniform_int_distribution<int> value_dist(-4000, 3999);
    for (int64_t i = 0; i < m; ++i) {
        idx[i] = static_cast<int32_t>(rng() % n);
        // Multiples of 1/4 with small magnitude add exactly in any order.
        values[i] = value_dist(rng) / 4.0;
    }
    std::vector<double> expected(n, 0.0);
    for (int64_t i = 0; i < m; ++i) {
        expected[idx[i]] += values[i];
    }

    int32_t* d_idx = nullptr;
    double* d_values = nullptr;
    double* d_out = nullptr;
    CHECK(cudaMalloc(&d_idx, (m + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_values, (m + 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
    CHECK(cudaMemcpy(d_idx, idx.data(), m * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, (n + 1) * sizeof(double)));
    if (m > 0) {
        const unsigned blocks = static_cast<unsigned>((m + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_idx, d_values, d_out, m, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<double> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_idx));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_out));

    for (int64_t k = 0; k < n; ++k) {
        if (got[k] != expected[k]) {
            std::fprintf(stderr, "m=%lld n=%lld: out[%lld] = %.17g, expected %.17g\n",
                         (long long)m, (long long)n, (long long)k, got[k], expected[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(9);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t m : sizes) {
        for (int64_t n : sizes) {
            if (m > 0 && n == 0) {
                continue;
            }
            ok = run_case(m, n, rng) && ok;
        }
    }
    std::printf("scatter_add cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
