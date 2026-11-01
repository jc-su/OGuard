#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

static int failures = 0;
static const int64_t LIM = (int64_t)1 << 61;

static int64_t draw(std::mt19937_64& rng, int64_t lo, int64_t hi) {
    return lo + (int64_t)(rng() % (uint64_t)(hi - lo + 1));
}

// kind: 0 wide, 1 near (small range, small tolerance), 2 zero tolerance,
// 3 maximum tolerance, 4 all equal.
static void run_case(std::mt19937_64& rng, int64_t n, int64_t r, int kind) {
    int64_t lo = -LIM, hi = LIM, tolerance = (int64_t)1 << 60;
    if (kind == 1) { lo = -1000; hi = 1000; tolerance = 5; }
    if (kind == 2) { lo = -100; hi = 100; tolerance = 0; }
    if (kind == 3) tolerance = (int64_t)1 << 62;
    if (kind == 4) { lo = 7; hi = 7; tolerance = 0; }
    std::vector<int64_t> left(n), right(r);
    for (auto& v : left) v = draw(rng, kind == 1 ? -1100 : lo, kind == 1 ? 1100 : hi);
    for (auto& v : right) v = draw(rng, lo, hi);
    if (kind == 3) {
        if (n > 0) left[0] = -LIM;
        if (n > 1) left[1] = LIM;
        if (r > 0) right[0] = LIM;
        if (r > 1) right[1] = -LIM;
    }
    std::sort(right.begin(), right.end());
    std::vector<int32_t> want(n), got(n, -5);
    for (int64_t i = 0; i < n; i++) {
        int64_t j = std::upper_bound(right.begin(), right.end(), left[i]) - right.begin() - 1;
        want[i] = (j >= 0 && left[i] - right[j] <= tolerance) ? (int32_t)j : -1;
    }
    int64_t *d_left, *d_right; int32_t* d_out;
    CHECK(cudaMalloc(&d_left, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_right, (r > 0 ? r : 1) * 8));
    CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * 4));
    if (n > 0) {
        CHECK(cudaMemcpy(d_left, left.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_out, got.data(), n * 4, cudaMemcpyHostToDevice));
    }
    if (r > 0) CHECK(cudaMemcpy(d_right, right.data(), r * 8, cudaMemcpyHostToDevice));
    int64_t grid = (n + 255) / 256;
    if (grid < 1) grid = 1;
    kernel<<<(unsigned)grid, 256>>>(d_left, d_right, d_out, n, r, tolerance);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    if (n > 0) CHECK(cudaMemcpy(got.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    bool ok = got == want;
    if (!ok) failures++;
    printf("%s asof_join cuda n=%lld r=%lld kind=%d\n", ok ? "PASS" : "FAIL", (long long)n,
           (long long)r, kind);
    cudaFree(d_left); cudaFree(d_right); cudaFree(d_out);
}

int main() {
    std::mt19937_64 rng(4);
    for (int64_t n : {0, 1, 1000, 1000003})
        for (int64_t r : {0, 1, 1000, 1000003})
            for (int kind = 0; kind < 5; kind++) run_case(rng, n, r, kind);
    return failures ? 1 : 0;
}
