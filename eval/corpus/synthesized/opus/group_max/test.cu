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

static bool run_case(int64_t n, int64_t capacity, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int32_t> group_dist(0, static_cast<int32_t>(capacity - 1));
    std::vector<int32_t> group(n);
    std::vector<int64_t> value(n);
    for (int64_t i = 0; i < n; ++i) {
        group[i] = group_dist(rng);
        value[i] = static_cast<int64_t>(rng());
    }
    std::vector<int64_t> best(capacity, LLONG_MIN);
    std::vector<int64_t> expected = best;
    for (int64_t i = 0; i < n; ++i) {
        if (value[i] > expected[group[i]]) {
            expected[group[i]] = value[i];
        }
    }

    int32_t* d_group = nullptr;
    int64_t* d_value = nullptr;
    int64_t* d_best = nullptr;
    CHECK(cudaMalloc(&d_group, (n + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_value, (n + 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_best, capacity * sizeof(int64_t)));
    CHECK(cudaMemcpy(d_group, group.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_value, value.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_best, best.data(), capacity * sizeof(int64_t), cudaMemcpyHostToDevice));
    if (n > 0) {
        const unsigned blocks = static_cast<unsigned>((n + THREADS - 1) / THREADS);
        kernel<<<blocks, THREADS>>>(d_group, d_value, d_best, n, capacity);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> got(capacity);
    CHECK(cudaMemcpy(got.data(), d_best, capacity * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_group));
    CHECK(cudaFree(d_value));
    CHECK(cudaFree(d_best));

    for (int64_t k = 0; k < capacity; ++k) {
        if (got[k] != expected[k]) {
            std::fprintf(stderr, "n=%lld capacity=%lld: best[%lld] = %lld, expected %lld\n",
                         (long long)n, (long long)capacity, (long long)k, (long long)got[k],
                         (long long)expected[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    std::mt19937_64 rng(4);
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t capacities[] = {1, 17, 200000};
    bool ok = true;
    for (int64_t n : sizes) {
        for (int64_t capacity : capacities) {
            ok = run_case(n, capacity, rng) && ok;
        }
    }
    std::printf("group_max cuda: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
