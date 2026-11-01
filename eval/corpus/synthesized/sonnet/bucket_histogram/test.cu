#include <algorithm>
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

static bool check(int64_t n, int64_t b, int mode, int sms) {
    std::mt19937_64 rng(555 + n * 17 + b);
    std::uniform_real_distribution<double> u(-100.0, 100.0);
    std::vector<double> bounds(b), values(n);
    for (auto& x : bounds) x = std::floor(u(rng));  // integers, so duplicates and exact hits occur
    std::sort(bounds.begin(), bounds.end());
    for (int64_t i = 0; i < n; i++) {
        uint64_t r = rng() % 4;
        if (r == 0 && b > 0)
            values[i] = bounds[rng() % (uint64_t)b];  // equal to a bound
        else if (r == 1)
            values[i] = std::floor(u(rng)) * 1.2;
        else
            values[i] = u(rng) * 1.2;
    }
    if (n > 3) {
        values[0] = INFINITY;
        values[1] = -INFINITY;
        values[2] = 0.0;
    }
    std::vector<int32_t> ref(b + 1, 0), got(b + 1);
    for (int64_t i = 0; i < n; i++)
        ref[std::upper_bound(bounds.begin(), bounds.end(), values[i]) - bounds.begin()]++;
    double *dv, *db;
    int32_t* dc;
    CK(cudaMalloc(&dv, (n ? n : 1) * 8));
    CK(cudaMalloc(&db, (b ? b : 1) * 8));
    CK(cudaMalloc(&dc, (b + 1) * 4));
    CK(cudaMemcpy(dv, values.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, bounds.data(), b * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dc, 0, (b + 1) * 4));
    kernel<<<grid_for(n, mode, sms), 256>>>(dv, db, dc, n, b);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dc, (b + 1) * 4, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(db);
    cudaFree(dc);
    for (int64_t k = 0; k <= b; k++)
        if (got[k] != ref[k]) {
            printf("FAIL n=%lld b=%lld mode=%d at bin %lld: got %d want %d\n", (long long)n,
                   (long long)b, mode, (long long)k, got[k], ref[k]);
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
        for (int64_t b : {(int64_t)0, (int64_t)1, (int64_t)10, (int64_t)1023, (int64_t)1024,
                          (int64_t)5000})
            for (int mode = 0; mode < 3; mode++) ok &= check(n, b, mode, sms);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
