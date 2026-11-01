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

// kind: 0 small values, 1 full-range values (wraparound), 2 all INT64_MAX,
// 3 all null, 4 skewed groups.
static void run_case(std::mt19937_64& rng, int64_t n, int64_t g, int kind, int sms) {
    int64_t nbytes = (n + 7) / 8;
    std::vector<int32_t> group(n);
    std::vector<int64_t> value(n);
    std::vector<uint8_t> valid(nbytes);
    for (int64_t i = 0; i < n; i++) {
        group[i] = (int32_t)(rng() % (uint64_t)g);
        if (kind == 4 && rng() % 10 != 0) group[i] = 0;
        if (kind == 1) value[i] = (int64_t)rng();
        else if (kind == 2) value[i] = INT64_MAX;
        else value[i] = (int64_t)(rng() % 2001) - 1000;
    }
    for (int64_t b = 0; b < nbytes; b++) {
        valid[b] = (kind == 2 || kind == 4) ? 255 : (kind == 3 ? 0 : (uint8_t)rng());
    }
    std::vector<uint64_t> want_c(g, 0), want_s(g, 0);
    for (int64_t i = 0; i < n; i++) {
        if ((valid[i >> 3] >> (i & 7)) & 1) {
            want_c[group[i]]++;
            want_s[group[i]] += (uint64_t)value[i];
        }
    }
    int32_t* d_group; int64_t *d_value, *d_counts, *d_sums; uint8_t* d_valid;
    CHECK(cudaMalloc(&d_group, (n > 0 ? n : 1) * 4));
    CHECK(cudaMalloc(&d_value, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_valid, nbytes > 0 ? nbytes : 1));
    CHECK(cudaMalloc(&d_counts, g * 8));
    CHECK(cudaMalloc(&d_sums, g * 8));
    if (n > 0) {
        CHECK(cudaMemcpy(d_group, group.data(), n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_value, value.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_valid, valid.data(), nbytes, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemset(d_counts, 0, g * 8));
    CHECK(cudaMemset(d_sums, 0, g * 8));
    int64_t grid = (n + 255) / 256;
    if (grid > 8 * sms) grid = 8 * sms;
    if (grid < 1) grid = 1;
    kernel<<<(unsigned)grid, 256>>>(d_group, d_value, d_valid, d_counts, d_sums, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> got_c(g), got_s(g);
    CHECK(cudaMemcpy(got_c.data(), d_counts, g * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_s.data(), d_sums, g * 8, cudaMemcpyDeviceToHost));
    bool ok = got_c == want_c && got_s == want_s;
    if (!ok) failures++;
    printf("%s group_sum_valid cuda n=%lld g=%lld kind=%d\n", ok ? "PASS" : "FAIL",
           (long long)n, (long long)g, kind);
    cudaFree(d_group); cudaFree(d_value); cudaFree(d_valid); cudaFree(d_counts); cudaFree(d_sums);
}

int main() {
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    std::mt19937_64 rng(3);
    for (int64_t n : {0, 1, 1000, 1000003})
        for (int64_t g : {1, 7, 256, 1024})
            for (int kind = 0; kind < 5; kind++) run_case(rng, n, g, kind, sms);
    return failures ? 1 : 0;
}
