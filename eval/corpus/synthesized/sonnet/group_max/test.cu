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

static bool check(int64_t n, int64_t capacity, int mode, int sms) {
    std::mt19937_64 rng(4242 + n * 13 + capacity);
    std::vector<int32_t> group(n);
    std::vector<int64_t> value(n);
    for (auto& x : group) x = (int32_t)(rng() % (uint64_t)capacity);
    for (auto& x : value) x = (int64_t)rng();
    std::vector<int64_t> ref(capacity, INT64_MIN), got(capacity);
    for (int64_t i = 0; i < n; i++) ref[group[i]] = std::max(ref[group[i]], value[i]);
    int32_t* dg;
    int64_t *dv, *db;
    CK(cudaMalloc(&dg, (n ? n : 1) * 4));
    CK(cudaMalloc(&dv, (n ? n : 1) * 8));
    CK(cudaMalloc(&db, (capacity ? capacity : 1) * 8));
    CK(cudaMemcpy(dg, group.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, value.data(), n * 8, cudaMemcpyHostToDevice));
    std::vector<int64_t> init(capacity, INT64_MIN);
    CK(cudaMemcpy(db, init.data(), capacity * 8, cudaMemcpyHostToDevice));
    kernel<<<grid_for(n, mode, sms), 256>>>(dg, dv, db, n, capacity);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), db, capacity * 8, cudaMemcpyDeviceToHost));
    cudaFree(dg);
    cudaFree(dv);
    cudaFree(db);
    for (int64_t k = 0; k < capacity; k++)
        if (got[k] != ref[k]) {
            printf("FAIL n=%lld capacity=%lld mode=%d at %lld: got %lld want %lld\n", (long long)n,
                   (long long)capacity, mode, (long long)k, (long long)got[k], (long long)ref[k]);
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
        std::vector<int64_t> caps = {1, 7, 64, 1000, 1024, 1025, 5000, 1000003};
        if (n == 0) caps.push_back(0);
        for (int64_t c : caps)
            for (int mode = 0; mode < 3; mode++) ok &= check(n, c, mode, sms);
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
