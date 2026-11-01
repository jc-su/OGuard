#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
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

static bool check(int64_t n, int64_t offset, std::mt19937_64& rng)
{
    int64_t src_bytes = (offset + n + 7) / 8;
    int64_t out_bytes = (n + 7) / 8;
    std::vector<uint8_t> src(src_bytes);
    for (auto& c : src) c = (uint8_t)rng();
    std::vector<uint8_t> expected(out_bytes, 0);
    for (int64_t i = 0; i < n; i++) {
        int64_t p = offset + i;
        if ((src[p / 8] >> (p % 8)) & 1) expected[i / 8] |= (uint8_t)(1 << (i % 8));
    }
    uint8_t *d_src, *d_out;
    CK(cudaMalloc(&d_src, src_bytes > 0 ? src_bytes : 1));
    CK(cudaMalloc(&d_out, out_bytes > 0 ? out_bytes : 1));
    CK(cudaMemcpy(d_src, src.data(), src_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0xA5, out_bytes > 0 ? out_bytes : 1));
    int64_t grid = out_bytes > 0 ? (out_bytes + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_src, d_out, n, offset);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(out_bytes);
    CK(cudaMemcpy(got.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_src);
    cudaFree(d_out);
    bool ok = true;
    for (int64_t j = 0; j < out_bytes; j++)
        if (got[j] != expected[j]) {
            printf("  mismatch byte %lld: got %02x expected %02x\n", (long long)j, got[j], expected[j]);
            ok = false;
            break;
        }
    printf("n=%lld offset=%lld: %s\n", (long long)n, (long long)offset, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    std::mt19937_64 rng(7);
    bool ok = true;
    for (int64_t n : {0, 1, 1000, 1000003})
        for (int64_t off : {0, 1, 3, 7, 8, 9, 13, 64, 12345}) ok &= check(n, off, rng);
    for (int64_t n : {0, 1, 1000}) {
        ok &= check(n, 1LL << 30, rng);
        ok &= check(n, (1LL << 30) - 5, rng);
    }
    return ok ? 0 : 1;
}
