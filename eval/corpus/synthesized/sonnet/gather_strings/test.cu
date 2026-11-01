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

static int run_case(const char *name, const Column &col, const std::vector<int32_t> &idx) {
    int64_t n = (int64_t)col.offsets.size() - 1;
    int64_t m = (int64_t)idx.size();
    std::vector<int32_t> out_offsets(m + 1, 0);
    for (int64_t i = 0; i < m; i++) {
        out_offsets[i + 1] = out_offsets[i] + (col.offsets[idx[i] + 1] - col.offsets[idx[i]]);
    }
    std::vector<uint8_t> expected(out_offsets[m]);
    for (int64_t i = 0; i < m; i++) {
        for (int32_t k = 0; k < out_offsets[i + 1] - out_offsets[i]; k++) {
            expected[out_offsets[i] + k] = col.chars[col.offsets[idx[i]] + k];
        }
    }
    int32_t *d_offsets, *d_idx, *d_out_offsets;
    uint8_t *d_chars, *d_out_chars;
    CHECK(cudaMalloc(&d_offsets, (n + 1) * 4));
    CHECK(cudaMalloc(&d_chars, col.chars.size() + 1));
    CHECK(cudaMalloc(&d_idx, m * 4 + 1));
    CHECK(cudaMalloc(&d_out_offsets, (m + 1) * 4));
    CHECK(cudaMalloc(&d_out_chars, expected.size() + 1));
    CHECK(cudaMemcpy(d_offsets, col.offsets.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chars, col.chars.data(), col.chars.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_idx, idx.data(), m * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out_offsets, out_offsets.data(), (m + 1) * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out_chars, 0xAA, expected.size() + 1));
    int64_t grid = (m + 7) / 8 > 0 ? (m + 7) / 8 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_offsets, d_chars, d_idx, d_out_offsets, d_out_chars, n, m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(expected.size() + 1);
    CHECK(cudaMemcpy(got.data(), d_out_chars, expected.size() + 1, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < expected.size(); i++) ok = ok && got[i] == expected[i];
    ok = ok && got[expected.size()] == 0xAA;
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    cudaFree(d_offsets);
    cudaFree(d_chars);
    cudaFree(d_idx);
    cudaFree(d_out_offsets);
    cudaFree(d_out_chars);
    return ok ? 0 : 1;
}

int main() {
    std::mt19937_64 rng(1);
    int failed = 0;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        Column col = make_column(rng, random_lens(rng, n, 24, 0.2));
        const int64_t ms[] = {0, 1, 1000, n};
        for (int64_t m : ms) {
            if (n == 0 && m > 0) continue;
            std::vector<int32_t> idx(m);
            for (auto &v : idx) v = (int32_t)(rng() % n);
            char name[64];
            snprintf(name, sizeof name, "n=%lld m=%lld", (long long)n, (long long)m);
            failed += run_case(name, col, idx);
        }
    }
    {
        std::vector<int64_t> lens = random_lens(rng, 1000, 20, 0.3);
        lens[17] = 500000;
        Column col = make_column(rng, lens);
        std::vector<int32_t> idx(1000);
        for (auto &v : idx) v = (int32_t)(rng() % 1000);
        idx[3] = idx[400] = idx[999] = 17;
        failed += run_case("long string", col, idx);
    }
    {
        Column col = make_column(rng, std::vector<int64_t>(1000, 0));
        std::vector<int32_t> idx(1000);
        for (auto &v : idx) v = (int32_t)(rng() % 1000);
        failed += run_case("all empty", col, idx);
    }
    return failed ? 1 : 0;
}
