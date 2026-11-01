#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(e_));                               \
            exit(2);                                                       \
        }                                                                  \
    } while (0)

// Grid modes: 0 covers every row with one thread, 1 uses 3 blocks, 2 caps the grid at 4 blocks per SM.
static int grid_for(int64_t n, int mode, int sms) {
    int64_t b = (n + 255) / 256;
    if (mode == 1 && b > 3) b = 3;
    if (mode == 2 && b > 4 * sms) b = 4 * sms;
    return b < 1 ? 1 : (int)b;
}

static bool check(int64_t n, int mode, int sms) {
    std::mt19937_64 rng(10 + n);
    std::vector<int32_t> v(n), ref(n), got(n);
    for (auto& x : v) x = (int32_t)rng();
    for (int64_t base = 0; base < n; base += 1024) {
        const int64_t len = n - base < 1024 ? n - base : 1024;
        for (int64_t j = 0; j < len; j++) ref[base + j] = v[base + len - 1 - j];
    }
    int32_t *dv, *dout;
    CK(cudaMalloc(&dv, (n ? n : 1) * 4));
    CK(cudaMalloc(&dout, (n ? n : 1) * 4));
    CK(cudaMemcpy(dv, v.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0x33, (n ? n : 1) * 4));
    kernel<<<grid_for(n, mode, sms), 256>>>(dv, dout, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, n * 4, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(dout);
    for (int64_t i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            printf("FAIL n=%lld mode=%d at %lld: got %d want %d\n", (long long)n, mode,
                   (long long)i, got[i], ref[i]);
            return false;
        }
    return true;
}

int main() {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // Sizes at and around a tile boundary are added to the required sizes.
    const int64_t sizes[] = {0, 1, 1000, 1023, 1024, 1025, 2048, 3000, 1000003};
    bool ok = true;
    for (int64_t n : sizes)
        for (int mode = 0; mode < 3; mode++) ok &= check(n, mode, sms);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
