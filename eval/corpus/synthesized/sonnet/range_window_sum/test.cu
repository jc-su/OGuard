#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include "kernel.cu"

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            exit(2);                                                       \
        }                                                                  \
    } while (0)

static const int64_t B62 = 1LL << 62;

static std::vector<int64_t> make_ts(int64_t n, const std::string& name, std::mt19937_64& rng)
{
    std::vector<int64_t> ts(n);
    if (name == "dense") {
        int64_t c = -1000;
        for (int64_t i = 0; i < n; i++) { c += rng() % 4; ts[i] = c; }
    } else if (name == "all_equal") {
        for (auto& x : ts) x = -12345;
    } else if (name == "wide") {
        for (auto& x : ts) x = (int64_t)(rng() % ((uint64_t)B62 * 2 + 1)) - B62;
        std::sort(ts.begin(), ts.end());
    } else if (name == "groups") {
        for (int64_t i = 0; i < n; i++) ts[i] = (i / 3000) * 2 - 5;
    } else if (name == "extreme") {
        for (auto& x : ts) x = (rng() & 1) ? B62 : -B62;
        std::sort(ts.begin(), ts.end());
    } else {
        int64_t c = 0;
        for (int64_t i = 0; i < n; i++) { c += (rng() % 1000 == 0) ? (1LL << 38) : (int64_t)(rng() % 3); ts[i] = c; }
    }
    return ts;
}

static bool check(int64_t n, const std::string& name, int64_t range_, std::mt19937_64& rng)
{
    std::vector<int64_t> ts = make_ts(n, name, rng);
    std::vector<int64_t> values(n);
    for (auto& v : values) v = (int64_t)rng();
    std::vector<uint64_t> prefix(n + 1, 0);
    for (int64_t i = 0; i < n; i++) prefix[i + 1] = prefix[i] + (uint64_t)values[i];
    std::vector<int64_t> expected(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t lo = std::lower_bound(ts.begin(), ts.end(), ts[i] - range_) - ts.begin();
        int64_t hi = std::upper_bound(ts.begin(), ts.end(), ts[i]) - ts.begin();
        expected[i] = (int64_t)(prefix[hi] - prefix[lo]);
    }
    int64_t *d_ts, *d_values, *d_out;
    CK(cudaMalloc(&d_ts, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&d_values, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&d_out, (n > 0 ? n : 1) * 8));
    CK(cudaMemcpy(d_ts, ts.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0x63, (n > 0 ? n : 1) * 8));
    kernel<<<1, 256>>>(d_ts, d_values, d_out, n, range_);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int64_t> got(n);
    CK(cudaMemcpy(got.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_ts);
    cudaFree(d_values);
    cudaFree(d_out);
    bool ok = true;
    for (int64_t i = 0; i < n; i++)
        if (got[i] != expected[i]) {
            printf("  mismatch row %lld: got %lld expected %lld\n", (long long)i, (long long)got[i],
                   (long long)expected[i]);
            ok = false;
            break;
        }
    printf("n=%lld ts=%s range=%lld: %s\n", (long long)n, name.c_str(), (long long)range_, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    std::mt19937_64 rng(15);
    bool ok = true;
    struct Case { const char* name; std::vector<int64_t> ranges; };
    std::vector<Case> cases = {
        {"dense", {5, 100, 5000}},       {"all_equal", {0, 10, B62}}, {"wide", {0, B62 / 2, B62}},
        {"groups", {0, 1, 3000}},        {"extreme", {0, B62}},       {"gaps", {0, 7, 1LL << 40}},
    };
    for (int64_t n : {0, 1, 1000, 2048, 2049, 1000003})
        for (const Case& c : cases)
            for (int64_t r : c.ranges) ok &= check(n, c.name, r, rng);
    return ok ? 0 : 1;
}
