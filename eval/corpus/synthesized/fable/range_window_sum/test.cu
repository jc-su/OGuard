// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "kernel.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

template <class T> static T* upload(const std::vector<T>& v) {
    T* d; CHECK(cudaMalloc(&d, (v.size() + 1) * sizeof(T)));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static bool run_case(long long n, int variant, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<long long> ts(n), values(n), out(n, 7), expect(n);
    long long range = 0;
    for (long long i = 0; i < n; ++i) values[i] = (long long)rng();
    long long t = 0;
    switch (variant) {
        case 0: for (long long i = 0; i < n; ++i) { t += rng() % 4; ts[i] = t; } range = 5; break;
        case 1: for (long long i = 0; i < n; ++i) ts[i] = -(1ll << 62); range = 0; break;
        case 2: t = -(1ll << 61); for (long long i = 0; i < n; ++i) { t += 1 + rng() % 2; ts[i] = t; } range = 7000; break;
        default: for (long long i = 0; i < n; ++i) ts[i] = (long long)(rng() >> 1) - (1ll << 62); std::sort(ts.begin(), ts.end()); range = 1ll << 62; break;
    }
    std::vector<unsigned long long> prefix(n + 1, 0);
    for (long long i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + (unsigned long long)values[i];
    for (long long i = 0; i < n; ++i) {
        long long lo = std::lower_bound(ts.begin(), ts.end(), ts[i] - range) - ts.begin();
        long long hi = std::upper_bound(ts.begin(), ts.end(), ts[i]) - ts.begin();
        expect[i] = (long long)(prefix[hi] - prefix[lo]);
    }
    long long* d_ts = upload(ts); long long* d_values = upload(values); long long* d_out = upload(out);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_ts, d_values, d_out, n, range);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld variant=%d row %lld: %lld != %lld\n", n, variant, i, out[i], expect[i]); }
    cudaFree(d_ts); cudaFree(d_values); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 4; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("range_window_sum cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
