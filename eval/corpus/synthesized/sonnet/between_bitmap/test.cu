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

static void run_case(std::mt19937_64& rng, int64_t n, const char* name, int kind) {
    int64_t nbytes = (n + 7) / 8;
    std::vector<int64_t> values(n);
    std::vector<uint8_t> valid(nbytes), want(nbytes, 0), got(nbytes, 0xA5);
    int64_t lo = 0, hi = 0;
    for (int64_t i = 0; i < n; i++) {
        switch (kind) {
            case 0: values[i] = (int64_t)(rng() % 100) - 50; break;
            case 1: values[i] = (int64_t)rng(); break;
            case 2: values[i] = (int64_t)(rng() % 6) - 3; break;
            default: values[i] = (int64_t)(rng() % 100) - 50; break;
        }
    }
    if (kind == 0) { lo = -10; hi = 20; }
    if (kind == 1) { lo = INT64_MIN; hi = INT64_MAX; if (n > 0) values[0] = INT64_MIN; if (n > 1) values[1] = INT64_MAX; }
    if (kind == 2) { lo = 1; hi = 1; }
    if (kind == 3) { lo = 10; hi = -10; }
    // Padding bits of the last validity byte are random, to check the output mask.
    for (int64_t b = 0; b < nbytes; b++) valid[b] = (uint8_t)rng();
    for (int64_t i = 0; i < n; i++) {
        bool v = (valid[i >> 3] >> (i & 7)) & 1;
        if (v && values[i] >= lo && values[i] <= hi) want[i >> 3] |= (uint8_t)(1 << (i & 7));
    }
    int64_t *d_values; uint8_t *d_valid, *d_out;
    CHECK(cudaMalloc(&d_values, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_valid, nbytes > 0 ? nbytes : 1));
    CHECK(cudaMalloc(&d_out, nbytes > 0 ? nbytes : 1));
    if (n > 0) CHECK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
    if (nbytes > 0) {
        CHECK(cudaMemcpy(d_valid, valid.data(), nbytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_out, got.data(), nbytes, cudaMemcpyHostToDevice));
    }
    int64_t grid = (n + 255) / 256;
    if (grid < 1) grid = 1;
    kernel<<<(unsigned)grid, 256>>>(d_values, d_valid, d_out, n, lo, hi);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    if (nbytes > 0) CHECK(cudaMemcpy(got.data(), d_out, nbytes, cudaMemcpyDeviceToHost));
    bool ok = got == want;
    if (!ok) failures++;
    printf("%s between_bitmap cuda n=%lld %s\n", ok ? "PASS" : "FAIL", (long long)n, name);
    cudaFree(d_values); cudaFree(d_valid); cudaFree(d_out);
}

int main() {
    std::mt19937_64 rng(1);
    const char* names[] = {"small", "full_range_extremes", "point", "empty_range"};
    for (int64_t n : {0, 1, 7, 8, 9, 1000, 1000003})
        for (int kind = 0; kind < 4; kind++) run_case(rng, n, names[kind], kind);
    return failures ? 1 : 0;
}
