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

struct Column {
    std::vector<int32_t> offsets;
    std::vector<uint8_t> chars;
};

static Column make_column(std::mt19937_64 &rng, const std::vector<int64_t> &lens) {
    Column c;
    c.offsets.assign(lens.size() + 1, 0);
    for (size_t i = 0; i < lens.size(); i++) c.offsets[i + 1] = c.offsets[i] + (int32_t)lens[i];
    c.chars.resize(c.offsets.back());
    for (auto &b : c.chars) b = (uint8_t)rng();
    return c;
}

static std::vector<int64_t> random_lens(std::mt19937_64 &rng, int64_t n, int max_len, double empty_frac) {
    std::vector<int64_t> lens(n);
    for (auto &l : lens) {
        l = (double)(rng() % 1000) / 1000.0 < empty_frac ? 0 : (int64_t)(rng() % (max_len + 1));
    }
    return lens;
}

template <typename T>
static T *upload(const std::vector<T> &v) {
    T *d;
    CHECK(cudaMalloc(&d, v.size() * sizeof(T) + 1));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static int run_case(const char *name, const Column &a, const Column &b) {
    int64_t n = (int64_t)a.offsets.size() - 1;
    std::vector<int32_t> out_offsets(n + 1);
    for (int64_t i = 0; i <= n; i++) out_offsets[i] = a.offsets[i] + b.offsets[i];
    std::vector<uint8_t> expected;
    for (int64_t i = 0; i < n; i++) {
        expected.insert(expected.end(), a.chars.begin() + a.offsets[i], a.chars.begin() + a.offsets[i + 1]);
        expected.insert(expected.end(), b.chars.begin() + b.offsets[i], b.chars.begin() + b.offsets[i + 1]);
    }
    int32_t *d_a_off = upload(a.offsets), *d_b_off = upload(b.offsets), *d_out_off = upload(out_offsets);
    uint8_t *d_a = upload(a.chars), *d_b = upload(b.chars);
    uint8_t *d_out;
    CHECK(cudaMalloc(&d_out, expected.size() + 1));
    CHECK(cudaMemset(d_out, 0xAA, expected.size() + 1));
    int64_t grid = (n + 7) / 8 > 0 ? (n + 7) / 8 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_a_off, d_a, d_b_off, d_b, d_out_off, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(expected.size() + 1);
    CHECK(cudaMemcpy(got.data(), d_out, expected.size() + 1, cudaMemcpyDeviceToHost));
    bool ok = got[expected.size()] == 0xAA;
    for (size_t i = 0; i < expected.size(); i++) ok = ok && got[i] == expected[i];
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    cudaFree(d_a_off);
    cudaFree(d_b_off);
    cudaFree(d_out_off);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return ok ? 0 : 1;
}

int main() {
    std::mt19937_64 rng(2);
    int failed = 0;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        Column a = make_column(rng, random_lens(rng, n, 24, 0.2));
        Column b = make_column(rng, random_lens(rng, n, 24, 0.2));
        char name[64];
        snprintf(name, sizeof name, "n=%lld", (long long)n);
        failed += run_case(name, a, b);
    }
    {
        std::vector<int64_t> la = random_lens(rng, 1000, 20, 0.3), lb = random_lens(rng, 1000, 20, 0.3);
        la[17] = 500000;
        lb[300] = 300000;
        la[300] = 0;
        lb[17] = 0;
        failed += run_case("long strings", make_column(rng, la), make_column(rng, lb));
    }
    std::vector<int64_t> zeros(1000, 0);
    failed += run_case("all empty", make_column(rng, zeros), make_column(rng, zeros));
    failed += run_case("a empty", make_column(rng, zeros), make_column(rng, random_lens(rng, 1000, 40, 0.1)));
    failed += run_case("b empty", make_column(rng, random_lens(rng, 1000, 40, 0.1)), make_column(rng, zeros));
    return failed ? 1 : 0;
}
