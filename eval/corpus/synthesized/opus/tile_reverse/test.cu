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
static const int64_t TILE = 1024;

static bool run_case(int64_t n, std::mt19937_64& rng)
{
    std::vector<int32_t> values(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = static_cast<int32_t>(rng());
    }
    std::vector<int32_t> expected(n);
    for (int64_t start = 0; start < n; start += TILE) {
        const int64_t end = start + TILE < n ? start + TILE : n;
        for (int64_t i = start; i < end; ++i) {
            expected[start + end - 1 - i] = values[i];
        }
    }

    int32_t* d_values = nullptr;
    int32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, (n + 1) * sizeof(int32_t)));
    if (n > 0) {
        const unsigned blocks = static_cast<unsigned>((n + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_values, d_out, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_out));

    for (int64_t i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            std::fprintf(stderr, "n=%lld: out[%lld] = %d, expected %d\n", (long long)n,
                         (long long)i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(10);
    const int64_t sizes[] = {0, 1, 1000, 1024, 2053, 1000003};
    bool ok = true;
    for (int64_t n : sizes) {
        ok = run_case(n, rng) && ok;
    }
    std::printf("tile_reverse cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
