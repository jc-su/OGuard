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

static bool check(int64_t n, int64_t range, int mode, int sms) {
    std::mt19937_64 rng(12345 + n);
    std::vector<int64_t> v(n);
    for (auto& x : v) x = (int64_t)(rng() % (uint64_t)range);
    std::vector<int8_t> ref(n), got(n);
    for (int64_t i = 0; i < n; i++) ref[i] = (i == 0 || v[i] != v[i - 1]) ? 1 : 0;
    int64_t* dv;
    int8_t* df;
    CK(cudaMalloc(&dv, (n ? n : 1) * 8));
    CK(cudaMalloc(&df, n ? n : 1));
    CK(cudaMemcpy(dv, v.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(df, 0x7f, n ? n : 1));
    kernel<<<grid_for(n, mode, sms), 256>>>(dv, df, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), df, n, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(df);
    for (int64_t i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            printf("FAIL n=%lld range=%lld mode=%d at %lld: got %d want %d\n", (long long)n,
                   (long long)range, mode, (long long)i, got[i], ref[i]);
            return false;
        }
    return true;
}

int main() {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    const int64_t ranges[] = {1, 3, 1000000000000LL};
    bool ok = true;
    for (int64_t n : sizes)
        for (int64_t r : ranges)
            for (int mode = 0; mode < 3; mode++) ok &= check(n, r, mode, sms);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
