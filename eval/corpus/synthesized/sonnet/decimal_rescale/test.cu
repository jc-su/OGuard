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

typedef __int128 i128;

static std::vector<int64_t> make_values(int64_t n, int shift, std::mt19937_64& rng)
{
    std::vector<int64_t> v(n);
    for (int64_t i = 0; i < n; i++) {
        uint64_t b = rng();
        v[i] = (int64_t)(b >> (rng() % 64)) * ((rng() & 1) ? -1 : 1);
    }
    i128 p = 1;
    for (int k = 0; k < (shift < 0 ? -shift : shift); k++) p *= 10;
    i128 lim = (i128)INT64_MAX / p;
    std::vector<i128> sp = {0, 1, -1, INT64_MIN, INT64_MAX, (i128)INT64_MIN + 1, (i128)INT64_MAX - 1};
    for (int d = -1; d <= 1; d++) {
        sp.push_back(lim + d);
        sp.push_back(-lim + d);
        sp.push_back(-lim - 1 + d);
    }
    for (int t : {1, 3, 5, 7, 15, 25, 35}) {
        i128 h = (p % 2 == 0) ? (i128)t * p / 2 : (i128)t * p;
        for (int d = -1; d <= 1; d++) {
            sp.push_back(h + d);
            sp.push_back(-h + d);
        }
    }
    int64_t c = 0;
    for (i128 s : sp) {
        if (s < INT64_MIN || s > INT64_MAX) continue;
        if (c >= n) break;
        v[c++] = (int64_t)s;
    }
    return v;
}

static bool check(int64_t n, int shift, std::mt19937_64& rng)
{
    std::vector<int64_t> v = make_values(n, shift, rng);
    std::vector<int64_t> eo(n);
    std::vector<int8_t> ev(n);
    i128 p = 1;
    for (int k = 0; k < (shift < 0 ? -shift : shift); k++) p *= 10;
    for (int64_t i = 0; i < n; i++) {
        if (shift >= 0) {
            i128 prod = (i128)v[i] * p;
            bool o = prod > INT64_MAX || prod < INT64_MIN;
            eo[i] = o ? 0 : (int64_t)prod;
            ev[i] = o ? 1 : 0;
        } else {
            i128 mag = v[i] < 0 ? -(i128)v[i] : (i128)v[i];
            i128 q = (2 * mag + p) / (2 * p);
            eo[i] = (int64_t)(v[i] < 0 ? -q : q);
            ev[i] = 0;
        }
    }
    int64_t *dv, *dout;
    int8_t* dovf;
    CK(cudaMalloc(&dv, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&dout, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&dovf, (n > 0 ? n : 1)));
    CK(cudaMemcpy(dv, v.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0x55, (n > 0 ? n : 1) * 8));
    CK(cudaMemset(dovf, 0x55, (n > 0 ? n : 1)));
    int64_t grid = n > 0 ? (n + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(dv, dout, dovf, n, shift);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int64_t> go(n);
    std::vector<int8_t> gv(n);
    CK(cudaMemcpy(go.data(), dout, n * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gv.data(), dovf, n, cudaMemcpyDeviceToHost));
    cudaFree(dv);
    cudaFree(dout);
    cudaFree(dovf);
    bool ok = true;
    for (int64_t i = 0; i < n; i++)
        if (go[i] != eo[i] || gv[i] != ev[i]) {
            printf("  mismatch row %lld value %lld: got %lld/%d expected %lld/%d\n", (long long)i,
                   (long long)v[i], (long long)go[i], gv[i], (long long)eo[i], ev[i]);
            ok = false;
            break;
        }
    printf("n=%lld shift=%d: %s\n", (long long)n, shift, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    std::mt19937_64 rng(5);
    bool ok = true;
    for (int64_t n : {0, 1, 1000})
        for (int s = -18; s <= 18; s++) ok &= check(n, s, rng);
    for (int s = -18; s <= 18; s++) ok &= check(1000003, s, rng);
    return ok ? 0 : 1;
}
