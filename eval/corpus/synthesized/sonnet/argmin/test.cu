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

static int sms = 0;

static bool run_case(const char* name, const std::vector<int64_t>& values) {
    int64_t n = (int64_t)values.size();
    int32_t expect = -1;
    for (int64_t i = 0; i < n; i++)
        if (expect < 0 || values[i] < values[expect]) expect = (int32_t)i;

    int64_t* d_values;
    int32_t* d_row;
    CHECK(cudaMalloc(&d_values, std::max<int64_t>(n, 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_row, sizeof(int32_t)));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    int32_t init = -1;
    CHECK(cudaMemcpy(d_row, &init, sizeof(int32_t), cudaMemcpyHostToDevice));
    int64_t grid = std::max<int64_t>(1, std::min<int64_t>((n + 255) / 256, 8 * sms));
    kernel<<<(unsigned)grid, 256>>>(d_values, d_row, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    int32_t got;
    CHECK(cudaMemcpy(&got, d_row, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_row));
    bool ok = got == expect;
    printf("argmin cuda n=%lld %s: %s", (long long)n, name, ok ? "PASS\n" : "FAIL");
    if (!ok) printf(" got %d expect %d\n", got, expect);
    return ok;
}

int main() {
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    bool ok = true;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        std::vector<int64_t> v(n);
        for (auto& x : v) x = (int64_t)rnd();
        ok &= run_case("random", v);
        for (auto& x : v) x = (int64_t)(rnd() % 7) - 3;
        ok &= run_case("ties", v);
        for (auto& x : v) x = 7;
        ok &= run_case("equal", v);
        for (auto& x : v) x = INT64_MAX;
        ok &= run_case("max", v);
        for (auto& x : v) x = INT64_MIN;
        ok &= run_case("min", v);
        for (int64_t i = 0; i < n; i++) v[i] = n - i;
        ok &= run_case("descending", v);
        for (int64_t i = 0; i < n; i++) v[i] = i;
        ok &= run_case("ascending", v);
        for (int64_t i = 0; i < n; i++) v[i] = (int64_t)(rnd() % 1000) + 1;
        if (n) v[n - 1] = 0;
        ok &= run_case("min_last", v);
        for (auto& x : v) x = INT64_MAX;
        if (n) { v[n / 2] = INT64_MIN; v[n - 1] = INT64_MIN; }
        ok &= run_case("two_min", v);
    }
    return ok ? 0 : 1;
}
