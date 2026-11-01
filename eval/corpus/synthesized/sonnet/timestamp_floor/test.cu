#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err_));                                 \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

static const int64_t LIMIT = 1ll << 62;

// Floor division with exact integer arithmetic.
static int64_t floor_to(int64_t t, int64_t unit) {
    int64_t q = t / unit;
    if (t % unit < 0) q -= 1;
    return q * unit;
}

static int run_case(const char *name, const std::vector<int64_t> &ts, int64_t unit) {
    int64_t n = (int64_t)ts.size();
    std::vector<int64_t> expected(n);
    for (int64_t i = 0; i < n; i++) expected[i] = floor_to(ts[i], unit);
    int64_t *d_ts, *d_out;
    CHECK(cudaMalloc(&d_ts, n * 8 + 1));
    CHECK(cudaMalloc(&d_out, n * 8 + 1));
    CHECK(cudaMemcpy(d_ts, ts.data(), n * 8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0x5A, n * 8 + 1));
    int64_t grid = (n + 255) / 256 > 0 ? (n + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_ts, d_out, n, unit);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    bool ok = got == expected;
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    cudaFree(d_ts);
    cudaFree(d_out);
    return ok ? 0 : 1;
}

static int64_t uniform(std::mt19937_64 &rng, int64_t lo, int64_t hi) {
    return lo + (int64_t)(rng() % (uint64_t)(hi - lo + 1));
}

int main() {
    std::mt19937_64 rng(4);
    int failed = 0;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t units[] = {1, 2, 7, 1000, 3600000000ll, 86400000000ll, 999999937, 1ll << 40, (1ll << 40) - 1, 1000003};
    for (int64_t n : sizes) {
        for (int64_t unit : units) {
            std::vector<int64_t> ts(n);
            for (auto &t : ts) t = uniform(rng, -LIMIT, LIMIT);
            if (n >= 8) {
                ts[0] = -LIMIT;
                ts[1] = LIMIT;
                ts[2] = 0;
                ts[3] = -1;
                ts[4] = uniform(rng, -LIMIT / unit + 1, LIMIT / unit - 1) * unit;
                ts[5] = uniform(rng, -LIMIT / unit + 1, LIMIT / unit - 1) * unit - 1;
                ts[6] = uniform(rng, -LIMIT / unit + 1, LIMIT / unit - 1) * unit + 1;
                ts[7] = uniform(rng, -LIMIT / unit + 1, LIMIT / unit - 1) * unit + unit - 1;
            }
            char name[96];
            snprintf(name, sizeof name, "n=%lld unit=%lld", (long long)n, (long long)unit);
            failed += run_case(name, ts, unit);
        }
    }
    const int64_t pre_units[] = {1, 7, 3600000000ll};
    for (int64_t unit : pre_units) {
        std::vector<int64_t> ts(1000);
        for (auto &t : ts) t = uniform(rng, -86400000000ll * 365 * 60, -1);
        char name[96];
        snprintf(name, sizeof name, "pre-1970 unit=%lld", (long long)unit);
        failed += run_case(name, ts, unit);
    }
    return failed ? 1 : 0;
}
