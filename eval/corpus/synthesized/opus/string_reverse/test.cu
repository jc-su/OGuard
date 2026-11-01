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

static bool run_case(int64_t n, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int32_t> len(0, 20);
    std::vector<int32_t> offsets(n + 1);
    offsets[0] = 0;
    for (int64_t r = 0; r < n; ++r) {
        offsets[r + 1] = offsets[r] + (r == 1 ? 3000 : len(rng));
    }
    const int64_t total = offsets[n];
    std::vector<uint8_t> chars(total);
    for (int64_t p = 0; p < total; ++p) {
        chars[p] = static_cast<uint8_t>(rng());
    }
    std::vector<uint8_t> expected(total);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t start = offsets[r];
        const int64_t end = offsets[r + 1];
        for (int64_t p = start; p < end; ++p) {
            expected[start + end - 1 - p] = chars[p];
        }
    }

    int32_t* d_offsets = nullptr;
    uint8_t* d_chars = nullptr;
    uint8_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_chars, total + 1));
    CHECK(cudaMalloc(&d_out, total + 1));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int32_t),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chars, chars.data(), total, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, total + 1));
    const int64_t bytes = offsets[n] - offsets[0];
    if (bytes > 0) {
        const unsigned blocks = static_cast<unsigned>((bytes + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_offsets, d_chars, d_out, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(total);
    CHECK(cudaMemcpy(got.data(), d_out, total, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_out));

    for (int64_t p = 0; p < total; ++p) {
        if (got[p] != expected[p]) {
            std::fprintf(stderr, "n=%lld: out[%lld] = %d, expected %d\n", (long long)n,
                         (long long)p, got[p], expected[p]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(5);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t n : sizes) {
        ok = run_case(n, rng) && ok;
    }
    std::printf("string_reverse cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
