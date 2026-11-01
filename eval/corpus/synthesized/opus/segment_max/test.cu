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

static bool run_case(int64_t g, std::mt19937_64& rng)
{
    std::vector<int32_t> offsets(g + 1);
    std::uniform_int_distribution<int32_t> len(0, 8);
    offsets[0] = 0;
    const int64_t long_seg = g > 0 ? static_cast<int64_t>(rng() % g) : -1;
    for (int64_t j = 0; j < g; ++j) {
        offsets[j + 1] = offsets[j] + (j == long_seg ? 5000 : len(rng));
    }
    const int64_t n = offsets[g];
    std::vector<int64_t> values(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = static_cast<int64_t>(rng());
    }
    std::vector<int64_t> expected(g, LLONG_MIN);
    for (int64_t j = 0; j < g; ++j) {
        for (int64_t i = offsets[j]; i < offsets[j + 1]; ++i) {
            expected[j] = values[i] > expected[j] ? values[i] : expected[j];
        }
    }

    int32_t* d_offsets = nullptr;
    int64_t* d_values = nullptr;
    int64_t* d_maxes = nullptr;
    CHECK(cudaMalloc(&d_offsets, (g + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_maxes, (g + 1) * sizeof(int64_t)));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (g + 1) * sizeof(int32_t),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_maxes, 5, (g + 1) * sizeof(int64_t)));
    if (g > 0) {
        const int64_t warps_per_block = THREADS / 32;
        const unsigned blocks = static_cast<unsigned>((g + warps_per_block - 1) / warps_per_block);
        kernel<<<blocks, THREADS>>>(d_offsets, d_values, d_maxes, g);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> got(g);
    CHECK(cudaMemcpy(got.data(), d_maxes, g * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_maxes));

    for (int64_t j = 0; j < g; ++j) {
        if (got[j] != expected[j]) {
            std::fprintf(stderr, "g=%lld: maxes[%lld] = %lld, expected %lld\n", (long long)g,
                         (long long)j, (long long)got[j], (long long)expected[j]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(2);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t g : sizes) {
        ok = run_case(g, rng) && ok;
    }
    std::printf("segment_max cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
