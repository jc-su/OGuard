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

// Grid modes: 0 gives every byte a thread, 1 uses 3 blocks, 2 caps the grid at 4 blocks per SM.
static int grid_for(int64_t bytes, int mode, int sms) {
    int64_t b = (bytes + 255) / 256;
    if (mode == 1 && b > 3) b = 3;
    if (mode == 2 && b > 4 * sms) b = 4 * sms;
    return b < 1 ? 1 : (int)b;
}

static bool check(int64_t n, int max_len, int mode, int sms) {
    std::mt19937_64 rng(31337 + n * 5 + max_len);
    std::vector<int32_t> off(n + 1);
    off[0] = 0;
    for (int64_t i = 0; i < n; i++) off[i + 1] = off[i] + (int32_t)(rng() % (uint64_t)(max_len + 1));
    const int64_t total = off[n];
    std::vector<uint8_t> chars(total), ref(total), got(total);
    for (auto& c : chars) c = (uint8_t)rng();
    for (int64_t i = 0; i < n; i++)
        for (int32_t k = off[i]; k < off[i + 1]; k++) ref[k] = chars[off[i] + off[i + 1] - 1 - k];
    int32_t* doff;
    uint8_t *dc, *dout;
    CK(cudaMalloc(&doff, (n + 1) * 4));
    CK(cudaMalloc(&dc, total ? total : 1));
    CK(cudaMalloc(&dout, total ? total : 1));
    CK(cudaMemcpy(doff, off.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, chars.data(), total, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0xaa, total ? total : 1));
    kernel<<<grid_for(total, mode, sms), 256>>>(doff, dc, dout, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, total, cudaMemcpyDeviceToHost));
    cudaFree(doff);
    cudaFree(dc);
    cudaFree(dout);
    for (int64_t p = 0; p < total; p++)
        if (got[p] != ref[p]) {
            printf("FAIL n=%lld max_len=%d mode=%d at byte %lld: got %d want %d\n", (long long)n,
                   max_len, mode, (long long)p, got[p], ref[p]);
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
        std::vector<int> lens = {0, 1, 20, 100};
        if (n <= 1000) lens.push_back(5000);
        for (int len : lens)
            for (int mode = 0; mode < 3; mode++) ok &= check(n, len, mode, sms);
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
