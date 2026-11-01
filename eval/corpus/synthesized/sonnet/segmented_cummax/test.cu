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

// kind: 0 single partition random, 1 single increasing, 2 single decreasing,
// 3 each row its own partition, 4 short partitions, 5 long partitions,
// 6 all INT64_MIN.
static void run_case(std::mt19937_64& rng, int64_t n, int kind) {
    std::vector<int64_t> values(n);
    std::vector<int32_t> first(n, 0);
    double p_start = kind == 3 ? 1.0 : (kind == 4 ? 0.1 : (kind == 5 ? 0.0003 : 0.0));
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int64_t i = 0; i < n; i++) {
        if (i == 0) first[i] = 0;
        else first[i] = (uni(rng) < p_start) ? (int32_t)i : first[i - 1];
        switch (kind) {
            case 1: values[i] = i - n / 2; break;
            case 2: values[i] = n - i; break;
            case 4: values[i] = (int64_t)(rng() % 200) - 100; break;
            case 6: values[i] = INT64_MIN; break;
            default: values[i] = (int64_t)rng(); break;
        }
    }
    std::vector<int64_t> want(n), got(n, 5);
    for (int64_t i = 0; i < n; i++)
        want[i] = first[i] == i ? values[i] : std::max(want[i - 1], values[i]);
    int64_t *d_values, *d_out; int32_t* d_first;
    CHECK(cudaMalloc(&d_values, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_first, (n > 0 ? n : 1) * 4));
    CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * 8));
    if (n > 0) {
        CHECK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_first, first.data(), n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_out, got.data(), n * 8, cudaMemcpyHostToDevice));
    }
    kernel<<<1, 256>>>(d_values, d_first, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    if (n > 0) CHECK(cudaMemcpy(got.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    bool ok = got == want;
    if (!ok) failures++;
    printf("%s segmented_cummax cuda n=%lld kind=%d\n", ok ? "PASS" : "FAIL", (long long)n, kind);
    cudaFree(d_values); cudaFree(d_first); cudaFree(d_out);
}

int main() {
    std::mt19937_64 rng(12);
    for (int64_t n : {0, 1, 2, 255, 2048, 2049, 1000, 1000003})
        for (int kind = 0; kind < 7; kind++) run_case(rng, n, kind);
    return failures ? 1 : 0;
}
