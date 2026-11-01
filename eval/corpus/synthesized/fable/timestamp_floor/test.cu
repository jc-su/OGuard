// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
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

static long long floor_div_mul(long long t, long long unit) {
    __int128 q = (__int128)t / unit;
    if (((__int128)t % unit != 0) && (t < 0)) q -= 1;
    return (long long)(q * unit);
}

static bool run_case(long long n, long long unit, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<long long> ts(n), out(n, 7), expect(n);
    for (long long i = 0; i < n; ++i) {
        if (i % 2) ts[i] = (long long)(rng() % (1ull << 45)) - (1ll << 44);
        else ts[i] = (long long)(rng() % ((1ull << 63) + 1)) - (1ll << 62);
    }
    if (n > 0) ts[0] = -(1ll << 62);
    if (n > 1) ts[1] = 1ll << 62;
    for (long long i = 0; i < n; ++i) expect[i] = floor_div_mul(ts[i], unit);
    long long* d_ts = upload(ts); long long* d_out = upload(out);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_ts, d_out, n, unit);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld unit=%lld row %lld: %lld != %lld\n", n, unit, i, out[i], expect[i]); }
    cudaFree(d_ts); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long units[] = {1, 7, 3600000000ll, 86400000000ll, 1ll << 40};
    bool ok = true;
    for (long long n : sizes) for (long long u : units) ok = run_case(n, u, (unsigned)(n + u % 1000)) && ok;
    printf("timestamp_floor cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
