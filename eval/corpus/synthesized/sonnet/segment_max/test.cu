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

// Grid modes: 0 gives every segment its own warp, 1 uses 3 blocks, 2 caps the grid at 8 blocks per SM.
static int grid_for(int64_t g, int mode, int sms) {
    int64_t b = (g + 7) / 8;
    if (mode == 1 && b > 3) b = 3;
    if (mode == 2 && b > 8 * sms) b = 8 * sms;
    return b < 1 ? 1 : (int)b;
}

static bool check(int64_t n, int64_t g, int mode, int sms) {
    std::mt19937_64 rng(777 + n * 31 + g);
    std::vector<int64_t> v(n);
    for (auto& x : v) x = (int64_t)rng();
    std::vector<int32_t> off(g + 1);
    std::vector<int64_t> cuts(g > 0 ? g - 1 : 0);
    for (auto& c : cuts) c = n ? (int64_t)(rng() % (uint64_t)(n + 1)) : 0;
    std::sort(cuts.begin(), cuts.end());
    off[0] = 0;
    for (int64_t j = 1; j < g; j++) off[j] = (int32_t)cuts[j - 1];
    off[g] = (int32_t)n;
    std::vector<int64_t> ref(g), got(g);
    for (int64_t j = 0; j < g; j++) {
        int64_t m = INT64_MIN;
        for (int32_t k = off[j]; k < off[j + 1]; k++) m = std::max(m, v[k]);
        ref[j] = m;
    }
    int32_t* doff;
    int64_t *dv, *dm;
    CK(cudaMalloc(&doff, (g + 1) * 4));
    CK(cudaMalloc(&dv, (n ? n : 1) * 8));
    CK(cudaMalloc(&dm, (g ? g : 1) * 8));
    CK(cudaMemcpy(doff, off.data(), (g + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, v.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dm, 0x55, (g ? g : 1) * 8));
    kernel<<<grid_for(g, mode, sms), 256>>>(doff, dv, dm, g);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dm, g * 8, cudaMemcpyDeviceToHost));
    cudaFree(doff);
    cudaFree(dv);
    cudaFree(dm);
    for (int64_t j = 0; j < g; j++)
        if (got[j] != ref[j]) {
            printf("FAIL n=%lld g=%lld mode=%d at %lld: got %lld want %lld\n", (long long)n,
                   (long long)g, mode, (long long)j, (long long)got[j], (long long)ref[j]);
            return false;
        }
    return true;
}

int main() {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t n : sizes) {
        // g = 0 needs offsets[0] = n = 0, so it only applies to n = 0.
        std::vector<int64_t> gs = {1, 3, n / 10 + 1, 2 * n + 5};
        if (n == 0) gs.push_back(0);
        for (int64_t g : gs)
            for (int mode = 0; mode < 3; mode++) ok &= check(n, g, mode, sms);
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
