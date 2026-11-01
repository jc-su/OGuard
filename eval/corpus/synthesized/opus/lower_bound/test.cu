#include <algorithm>
#include <climits>
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

static bool run_case(int64_t n, int64_t s, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int64_t> key_dist(-1000, 999);
    std::uniform_int_distribution<int64_t> sorted_dist(-900, 899);
    std::vector<int64_t> keys(n);
    for (int64_t i = 0; i < n; ++i) {
        keys[i] = key_dist(rng);
    }
    if (n > 0) {
        keys[0] = LLONG_MIN;
        keys[n - 1] = LLONG_MAX;
    }
    std::vector<int64_t> sorted(s);
    for (int64_t i = 0; i < s; ++i) {
        sorted[i] = sorted_dist(rng);
    }
    std::sort(sorted.begin(), sorted.end());
    std::vector<int32_t> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        expected[i] = static_cast<int32_t>(
            std::lower_bound(sorted.begin(), sorted.end(), keys[i]) - sorted.begin());
    }

    int64_t* d_keys = nullptr;
    int64_t* d_sorted = nullptr;
    int32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_sorted, (s + 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sorted, sorted.data(), s * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0xff, (n + 1) * sizeof(int32_t)));
    if (n > 0) {
        const unsigned blocks = static_cast<unsigned>((n + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_keys, d_sorted, d_out, n, s);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_sorted));
    CHECK(cudaFree(d_out));

    for (int64_t i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            std::fprintf(stderr, "n=%lld s=%lld: out[%lld] = %d, expected %d\n", (long long)n,
                         (long long)s, (long long)i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(3);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t sorted_sizes[] = {0, 1, 37, 100000};
    bool ok = true;
    for (int64_t n : sizes) {
        for (int64_t s : sorted_sizes) {
            ok = run_case(n, s, rng) && ok;
        }
    }
    std::printf("lower_bound cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
