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

static bool run_case(const char* name, std::vector<int64_t> build, const std::vector<int64_t>& probe) {
    std::sort(build.begin(), build.end());
    int64_t b = (int64_t)build.size(), p = (int64_t)probe.size();
    std::vector<int32_t> offsets(p + 1, 0), exp_build, exp_probe;
    int64_t total = 0;
    for (int64_t i = 0; i < p; i++) {
        auto lo = std::lower_bound(build.begin(), build.end(), probe[i]);
        auto hi = std::upper_bound(build.begin(), build.end(), probe[i]);
        for (auto it = lo; it != hi; ++it) {
            exp_build.push_back((int32_t)(it - build.begin()));
            exp_probe.push_back((int32_t)i);
        }
        total += hi - lo;
        offsets[i + 1] = (int32_t)total;
    }

    int64_t *d_build, *d_probe;
    int32_t *d_offsets, *d_ob, *d_op;
    CHECK(cudaMalloc(&d_build, std::max<int64_t>(b, 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_probe, std::max<int64_t>(p, 1) * sizeof(int64_t)));
    CHECK(cudaMalloc(&d_offsets, (p + 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_ob, std::max<int64_t>(total, 1) * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_op, std::max<int64_t>(total, 1) * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_build, build.data(), b * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_probe, probe.data(), p * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (p + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_ob, 0xf9, std::max<int64_t>(total, 1) * sizeof(int32_t)));
    CHECK(cudaMemset(d_op, 0xf9, std::max<int64_t>(total, 1) * sizeof(int32_t)));
    int64_t grid = std::max<int64_t>(1, (p + 255) / 256);
    kernel<<<(unsigned)grid, 256>>>(d_build, d_probe, d_offsets, d_ob, d_op, b, p);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got_build(total), got_probe(total);
    CHECK(cudaMemcpy(got_build.data(), d_ob, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_probe.data(), d_op, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_build));
    CHECK(cudaFree(d_probe));
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_ob));
    CHECK(cudaFree(d_op));
    bool ok = got_build == exp_build && got_probe == exp_probe;
    printf("join_materialize cuda %s b=%lld p=%lld total=%lld: %s\n", name, (long long)b,
           (long long)p, (long long)total, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        std::vector<int64_t> build(n), probe(n);
        for (auto& x : build) x = (int64_t)(rnd() % (2 * n + 1)) - n;
        for (auto& x : probe) x = (int64_t)(rnd() % (2 * n + 1)) - n;
        ok &= run_case("unique", build, probe);
        ok &= run_case("empty_build", {}, probe);
        ok &= run_case("empty_probe", build, {});

        int64_t range = std::max<int64_t>(1, n / 10);
        for (auto& x : build) x = (int64_t)(rnd() % range);
        std::vector<int64_t> probe_dup(n ? range : 0);
        for (auto& x : probe_dup) x = (int64_t)(rnd() % (range + 2)) - 1;
        ok &= run_case("duplicates", build, probe_dup);

        // Every key has 40 matches, so every row takes the cooperative path.
        for (int64_t i = 0; i < n; i++) build[i] = i / 40;
        std::vector<int64_t> probe_wide(std::min<int64_t>(n, 20000));
        for (auto& x : probe_wide) x = (int64_t)(rnd() % std::max<int64_t>(1, n / 40));
        ok &= run_case("wide_rows", build, probe_wide);

        // One key with many matches among rows with none.
        for (int64_t i = 0; i < n; i++) build[i] = i < std::min<int64_t>(n, 5000) ? 5 : 100 + (int64_t)(rnd() % 100);
        std::vector<int64_t> probe_skew(std::min<int64_t>(n, 3000));
        for (auto& x : probe_skew) x = (int64_t)(rnd() % 10);
        ok &= run_case("skew", build, probe_skew);
    }
    ok &= run_case("extreme", {INT64_MAX, INT64_MIN, 0, INT64_MIN, INT64_MAX},
                   {INT64_MIN, INT64_MAX, 0, 1, INT64_MIN + 1});
    return ok ? 0 : 1;
}
