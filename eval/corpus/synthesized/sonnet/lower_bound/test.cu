#include <algorithm>
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

static bool check(int64_t n, int64_t s, int mode, int sms) {
    std::mt19937_64 rng(99 + n * 7 + s);
    std::vector<int64_t> sorted(s), keys(n);
    for (auto& x : sorted) x = (int64_t)(rng() % 2001) - 1000;  // many duplicates
    std::sort(sorted.begin(), sorted.end());
    for (auto& x : keys) x = (int64_t)(rng() % 2201) - 1100;    // hits, misses, and out of range
    if (n > 2) {
        keys[0] = INT64_MIN;
        keys[1] = INT64_MAX;
    }
    std::vector<int32_t> ref(n), got(n);
    for (int64_t i = 0; i < n; i++)
        ref[i] = (int32_t)(std::lower_bound(sorted.begin(), sorted.end(), keys[i]) - sorted.begin());
    int64_t *dk, *ds;
    int32_t* dout;
    CK(cudaMalloc(&dk, (n ? n : 1) * 8));
    CK(cudaMalloc(&ds, (s ? s : 1) * 8));
    CK(cudaMalloc(&dout, (n ? n : 1) * 4));
    CK(cudaMemcpy(dk, keys.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, sorted.data(), s * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0x7f, (n ? n : 1) * 4));
    kernel<<<grid_for(n, mode, sms), 256>>>(dk, ds, dout, n, s);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, n * 4, cudaMemcpyDeviceToHost));
    cudaFree(dk);
    cudaFree(ds);
    cudaFree(dout);
    for (int64_t i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            printf("FAIL n=%lld s=%lld mode=%d at %lld: got %d want %d\n", (long long)n,
                   (long long)s, mode, (long long)i, got[i], ref[i]);
            return false;
        }
    return true;
}

int main() {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t n : sizes)
        for (int64_t s : {(int64_t)0, (int64_t)1, (int64_t)1000, (int64_t)1000003})
            for (int mode = 0; mode < 3; mode++) ok &= check(n, s, mode, sms);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
