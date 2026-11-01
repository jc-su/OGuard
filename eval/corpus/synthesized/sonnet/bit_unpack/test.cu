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

// Reference packer: sets the bits of the stream one by one.
static std::vector<uint32_t> pack(const std::vector<uint32_t> &codes, int bits) {
    int64_t n = (int64_t)codes.size();
    std::vector<uint32_t> packed((n * bits + 31) / 32, 0);
    for (int64_t i = 0; i < n; i++) {
        for (int b = 0; b < bits; b++) {
            if ((codes[i] >> b) & 1) {
                int64_t pos = i * bits + b;
                packed[pos / 32] |= 1u << (pos % 32);
            }
        }
    }
    return packed;
}

static int run_case(const char *name, const std::vector<uint32_t> &codes, int bits) {
    int64_t n = (int64_t)codes.size();
    std::vector<uint32_t> packed = pack(codes, bits);
    int64_t w = (int64_t)packed.size();
    uint32_t *d_packed, *d_out;
    CHECK(cudaMalloc(&d_packed, w * 4 + 1));
    CHECK(cudaMalloc(&d_out, n * 4 + 1));
    CHECK(cudaMemcpy(d_packed, packed.data(), w * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0xFF, n * 4 + 1));
    int64_t grid = (n + 255) / 256 > 0 ? (n + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_packed, d_out, n, w, bits);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    bool ok = got == codes;
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    cudaFree(d_packed);
    cudaFree(d_out);
    return ok ? 0 : 1;
}

static std::vector<uint32_t> random_codes(std::mt19937_64 &rng, int64_t n, int bits) {
    std::vector<uint32_t> codes(n);
    uint64_t mask = (1ull << bits) - 1;
    for (auto &c : codes) c = (uint32_t)(rng() & mask);
    return codes;
}

int main() {
    std::mt19937_64 rng(3);
    int failed = 0;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int bit_list[] = {1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 32};
    for (int64_t n : sizes) {
        for (int bits : bit_list) {
            char name[64];
            snprintf(name, sizeof name, "n=%lld bits=%d", (long long)n, bits);
            failed += run_case(name, random_codes(rng, n, bits), bits);
        }
    }
    for (int bits = 1; bits <= 32; bits++) {
        char name[64];
        snprintf(name, sizeof name, "max codes bits=%d", bits);
        std::vector<uint32_t> maxcodes(1000, (uint32_t)((1ull << bits) - 1));
        failed += run_case(name, maxcodes, bits);
        snprintf(name, sizeof name, "random bits=%d", bits);
        failed += run_case(name, random_codes(rng, 1003, bits), bits);
    }
    return failed ? 1 : 0;
}
