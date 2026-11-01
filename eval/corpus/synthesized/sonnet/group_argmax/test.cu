#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "kernel.cu"

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(e_));                                 \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

static uint64_t state = 88172645463325252ULL;
static uint64_t rnd() {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return state;
}

static bool run_case(const char* name, int64_t g, const std::vector<int32_t>& group,
                     const std::vector<int64_t>& value) {
    int64_t n = (int64_t)group.size();
    std::vector<int32_t> expect(g, -1), got(g);
    for (int64_t i = 0; i < n; i++) {
        int32_t& b = expect[group[i]];
        if (b < 0 || value[i] > value[b]) b = (int32_t)i;
    }

    int32_t *d_group, *d_best;
    int64_t* d_value;
    CHECK(cudaMalloc(&d_group, std::max<int64_t>(n, 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_value, std::max<int64_t>(n, 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_best, g * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_group, group.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_value, value.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_best, 0xff, g * sizeof(int32_t)));
    int64_t grid = std::max<int64_t>(1, (n + 255) / 256);
    kernel<<<(unsigned)grid, 256>>>(d_group, d_value, d_best, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), d_best, g * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_group));
    CHECK(cudaFree(d_value));
    CHECK(cudaFree(d_best));
    bool ok = memcmp(got.data(), expect.data(), g * sizeof(int32_t)) == 0;
    printf("group_argmax cuda n=%lld g=%lld %s: %s\n", (long long)n, (long long)g, name,
           ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t groups[] = {1, 7, 1000, 1 << 20};
    for (int64_t n : sizes) {
        for (int64_t g : groups) {
            std::vector<int32_t> group(n);
            std::vector<int64_t> value(n);
            for (auto& x : group) x = (int32_t)(rnd() % (uint64_t)g);
            for (auto& x : value) x = (int64_t)rnd();
            ok &= run_case("random", g, group, value);
            for (auto& x : value) x = (int64_t)(rnd() % 5) - 2;
            ok &= run_case("ties", g, group, value);
            for (auto& x : value) x = INT64_MIN;
            ok &= run_case("equal", g, group, value);
            for (int64_t i = 0; i < n; i++) value[i] = i;
            ok &= run_case("ascending", g, group, value);
            for (int64_t i = 0; i < n; i++) value[i] = n - i;
            ok &= run_case("descending", g, group, value);
            for (auto& x : group) x = 0;
            for (auto& x : value) x = (int64_t)(rnd() % 11) - 5;
            ok &= run_case("sparse", g, group, value);
        }
    }
    return ok ? 0 : 1;
}
