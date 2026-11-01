#include <cmath>
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

// Grid modes: 0 covers every row with one thread, 1 uses 3 blocks, 2 caps the grid at 8 blocks per SM.
static int grid_for(int64_t m, int mode, int sms) {
    int64_t b = (m + 255) / 256;
    if (mode == 1 && b > 3) b = 3;
    if (mode == 2 && b > 8 * sms) b = 8 * sms;
    return b < 1 ? 1 : (int)b;
}

// Values are in [0, 1), so each sum has no cancellation and a relative tolerance bounds the
// difference caused by the order of the additions.
static bool check(int64_t m, int64_t n, int mode, int sms) {
    std::mt19937_64 rng(909 + m * 3 + n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int32_t> idx(m);
    std::vector<double> v(m);
    for (auto& x : idx) x = (int32_t)(rng() % (uint64_t)n);
    for (auto& x : v) x = u(rng);
    std::vector<long double> ref(n, 0.0L);
    for (int64_t i = 0; i < m; i++) ref[idx[i]] += v[i];
    std::vector<double> got(n);
    int32_t* di;
    double *dv, *dout;
    CK(cudaMalloc(&di, (m ? m : 1) * 4));
    CK(cudaMalloc(&dv, (m ? m : 1) * 8));
    CK(cudaMalloc(&dout, (n ? n : 1) * 8));
    CK(cudaMemcpy(di, idx.data(), m * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, v.data(), m * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0, (n ? n : 1) * 8));
    kernel<<<grid_for(m, mode, sms), 256>>>(di, dv, dout, m, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, n * 8, cudaMemcpyDeviceToHost));
    cudaFree(di);
    cudaFree(dv);
    cudaFree(dout);
    for (int64_t k = 0; k < n; k++)
        if (got[k] != (double)ref[k] && std::fabs(got[k] - (double)ref[k]) > 1e-9 * (double)ref[k]) {
            printf("FAIL m=%lld n=%lld mode=%d at %lld: got %.17g want %.17g\n", (long long)m,
                   (long long)n, mode, (long long)k, got[k], (double)ref[k]);
            return false;
        }
    return true;
}

int main() {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (int64_t m : sizes) {
        // out has at least one row whenever there is an index to scatter; n = 0 only pairs with m = 0.
        std::vector<int64_t> outs = {1, 10, 1000, 1000003};
        if (m == 0) outs.push_back(0);
        for (int64_t n : outs)
            for (int mode = 0; mode < 3; mode++) ok &= check(m, n, mode, sms);
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
