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

// Grid modes: 0 covers every row with one thread, 1 uses 3 blocks, 2 caps the grid at 4 blocks per SM.
static int grid_for(int64_t n, int mode, int sms) {
    int64_t b = (n + 255) / 256;
    if (mode == 1 && b > 3) b = 3;
    if (mode == 2 && b > 4 * sms) b = 4 * sms;
    return b < 1 ? 1 : (int)b;
}

// Values are in [0, 1), so the sum has no cancellation and a relative tolerance bounds the
// difference caused by the order of the additions.
static bool check(int64_t n, int keep_percent, int mode, int sms) {
    std::mt19937_64 rng(2024 + n * 3 + keep_percent);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> v(n);
    std::vector<int8_t> m(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = u(rng);
        m[i] = (int)(rng() % 100) < keep_percent ? (int8_t)(1 + rng() % 100) : 0;  // nonzero, not only 1
    }
    long double ref = 0;
    for (int64_t i = 0; i < n; i++)
        if (m[i] != 0) ref += v[i];
    double *dv, *dr;
    int8_t* dm;
    CK(cudaMalloc(&dv, (n ? n : 1) * 8));
    CK(cudaMalloc(&dm, n ? n : 1));
    CK(cudaMalloc(&dr, 8));
    CK(cudaMemcpy(dv, v.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm, m.data(), n, cudaMemcpyHostToDevice));
    CK(cudaMemset(dr, 0, 8));
    kernel<<<grid_for(n, mode, sms), 256>>>(dv, dm, dr, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    double got;
    CK(cudaMemcpy(&got, dr, 8, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(dm);
    cudaFree(dr);
    if (std::fabs(got - (double)ref) > 1e-9 * (double)ref + 1e-300 && got != (double)ref) {
        printf("FAIL n=%lld keep=%d mode=%d: got %.17g want %.17g\n", (long long)n, keep_percent,
               mode, got, (double)ref);
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
        for (int keep : {0, 1, 50, 100})
            for (int mode = 0; mode < 3; mode++) ok &= check(n, keep, mode, sms);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
