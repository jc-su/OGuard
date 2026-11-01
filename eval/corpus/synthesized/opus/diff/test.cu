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
    std::uniform_int_distribution<int64_t> dist(-(int64_t(1) << 40), int64_t(1) << 40);
    std::vector<int64_t> values(n);
    for (int64_t i = 0; i < n; ++i) {
        values[i] = dist(rng);
    }
    std::vector<int64_t> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        expected[i] = i == 0 ? values[0] : values[i] - values[i - 1];
    }

    int64_t* d_values = nullptr;
    int64_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int64_t)));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 3, (n + 1) * sizeof(int64_t)));
    if (n > 0) {
        const unsigned blocks = static_cast<unsigned>((n + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_values, d_out, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_out));

    for (int64_t i = 0; i < n; ++i) {
        if (got[i] != expected[i]) {
            std::fprintf(stderr, "n=%lld: out[%lld] = %lld, expected %lld\n", (long long)n,
                         (long long)i, (long long)got[i], (long long)expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(8);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t n : sizes) {
        ok = run_case(n, rng) && ok;
    }
    std::printf("diff cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
