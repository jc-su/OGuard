#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include "kernel.cu"

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            exit(2);                                                       \
        }                                                                  \
    } while (0)

static bool check(int64_t n, const std::string& mode, std::mt19937_64& rng)
{
    std::vector<int64_t> lens(n);
    for (int64_t i = 0; i < n; i++) {
        if (mode == "short") lens[i] = rng() % 17;
        else if (mode == "mixed") lens[i] = (rng() % 5000 == 0) ? 3000 : rng() % 12;
        else if (mode == "empty") lens[i] = 0;
        else lens[i] = (i == n / 2) ? 200000 : 0;
    }
    std::vector<int32_t> offsets(n + 1, 0);
    for (int64_t i = 0; i < n; i++) offsets[i + 1] = offsets[i] + (int32_t)lens[i];
    std::vector<uint8_t> chars(offsets[n]);
    for (auto& c : chars) c = (uint8_t)rng();
    std::vector<uint32_t> expected(n);
    for (int64_t i = 0; i < n; i++) {
        uint32_t h = 2166136261u;
        for (int32_t p = offsets[i]; p < offsets[i + 1]; p++) h = (h ^ chars[p]) * 16777619u;
        expected[i] = h;
    }
    int32_t* d_off;
    uint8_t* d_chars;
    uint32_t* d_out;
    CK(cudaMalloc(&d_off, (n + 1) * 4));
    CK(cudaMalloc(&d_chars, chars.size() > 0 ? chars.size() : 1));
    CK(cudaMalloc(&d_out, (n > 0 ? n : 1) * 4));
    CK(cudaMemcpy(d_off, offsets.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_chars, chars.data(), chars.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0xAB, (n > 0 ? n : 1) * 4));
    int64_t grid = n > 0 ? (n + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_off, d_chars, d_out, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(n);
    CK(cudaMemcpy(got.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_off);
    cudaFree(d_chars);
    cudaFree(d_out);
    bool ok = true;
    for (int64_t i = 0; i < n; i++)
        if (got[i] != expected[i]) {
            printf("  mismatch row %lld: got %u expected %u\n", (long long)i, got[i], expected[i]);
            ok = false;
            break;
        }
    printf("n=%lld mode=%s: %s\n", (long long)n, mode.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    std::mt19937_64 rng(6);
    bool ok = true;
    for (int64_t n : {0, 1, 1000, 1000003})
        for (const char* mode : {"short", "mixed", "empty", "one_long"}) ok &= check(n, mode, rng);
    return ok ? 0 : 1;
}
