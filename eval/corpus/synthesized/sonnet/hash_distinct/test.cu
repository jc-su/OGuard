#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <map>
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

static const int32_t ROW_EMPTY = 2147483647;

static int64_t start_slot(int64_t key, int64_t capacity) {
    return (int64_t)((((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32) & ((uint64_t)capacity - 1));
}

// Returns nullptr when the table meets the contract, else a description.
static const char* check(const std::vector<int64_t>& keys, const std::vector<int64_t>& tk,
                         const std::vector<int32_t>& tr, int64_t capacity) {
    std::map<int64_t, int32_t> first;
    for (int64_t i = 0; i < (int64_t)keys.size(); i++) first.emplace(keys[i], (int32_t)i);
    int64_t occupied = 0;
    for (int64_t s = 0; s < capacity; s++) {
        if (tk[s] == -1) {
            if (tr[s] != ROW_EMPTY) return "an empty slot has a row";
            continue;
        }
        occupied++;
        auto it = first.find(tk[s]);
        if (it == first.end()) return "a slot holds a key that is not in the input";
        if (tr[s] != it->second) return "a slot holds a row that is not the first of its key";
        int64_t p = start_slot(tk[s], capacity);
        while (p != s) {
            if (tk[p] == -1) return "a key is not reachable by probing";
            p = (p + 1) & (capacity - 1);
        }
    }
    if (occupied != (int64_t)first.size()) return "slots do not hold each distinct key once";
    return nullptr;
}

static bool run_case(const char* name, const std::vector<int64_t>& keys, int64_t capacity) {
    int64_t n = (int64_t)keys.size();
    int64_t *d_keys, *d_tk;
    int32_t* d_tr;
    CHECK(cudaMalloc(&d_keys, std::max<int64_t>(n, 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_tk, capacity * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_tr, capacity * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_tk, 0xff, capacity * sizeof(int64_t)));
    std::vector<int32_t> init(capacity, ROW_EMPTY);
    CHECK(cudaMemcpy(d_tr, init.data(), capacity * sizeof(int32_t), cudaMemcpyHostToDevice));
    int64_t grid = std::max<int64_t>(1, (n + 255) / 256);
    kernel<<<(unsigned)grid, 256>>>(d_keys, d_tk, d_tr, n, capacity);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> tk(capacity);
    std::vector<int32_t> tr(capacity);
    CHECK(cudaMemcpy(tk.data(), d_tk, capacity * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(tr.data(), d_tr, capacity * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_tk));
    CHECK(cudaFree(d_tr));
    const char* err = check(keys, tk, tr, capacity);
    printf("hash_distinct cuda n=%lld capacity=%lld %s: %s %s\n", (long long)n,
           (long long)capacity, name, err ? "FAIL" : "PASS", err ? err : "");
    return err == nullptr;
}

static int64_t capacity_for(int64_t n, int64_t factor) {
    int64_t c = 1;
    while (c < 2 * n * factor) c *= 2;
    return c;
}

int main() {
    bool ok = true;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        std::vector<int64_t> keys(n);
        for (int factor = 1; factor <= 4; factor *= 4) {
            int64_t cap = capacity_for(n, factor);
            for (auto& x : keys) { x = (int64_t)rnd(); if (x == -1) x = 0; }
            ok &= run_case("random", keys, cap);
            for (auto& x : keys) x = (int64_t)(rnd() % 10);
            ok &= run_case("few_distinct", keys, cap);
            for (auto& x : keys) x = -2 - (int64_t)(rnd() % 1000);
            ok &= run_case("negative", keys, cap);
            for (auto& x : keys) x = INT64_MIN;
            ok &= run_case("equal", keys, cap);
            for (auto& x : keys) x = (int64_t)(rnd() % (1ULL << 30)) << 32;
            ok &= run_case("high_words", keys, cap);
            for (int64_t i = 0; i < n; i++) keys[i] = i;
            ok &= run_case("sequential", keys, cap);
            for (auto& x : keys) x = (int64_t)(rnd() % (uint64_t)std::max<int64_t>(1, n / 2));
            ok &= run_case("distinct_half", keys, cap);
        }
    }
    return ok ? 0 : 1;
}
