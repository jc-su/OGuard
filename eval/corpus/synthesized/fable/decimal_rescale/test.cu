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

static void reference(long long v, long long shift, long long* out, signed char* ovf) {
    __int128 x = v;
    if (shift >= 0) {
        __int128 p = 1;
        for (long long k = 0; k < shift; ++k) p *= 10;
        __int128 r = x * p;
        if (r > (__int128)0x7FFFFFFFFFFFFFFFll || r < -(__int128)0x7FFFFFFFFFFFFFFFll - 1) { *out = 0; *ovf = 1; }
        else { *out = (long long)r; *ovf = 0; }
    } else {
        __int128 q = 1;
        for (long long k = 0; k < -shift; ++k) q *= 10;
        __int128 mag = x < 0 ? -x : x;
        __int128 qt = mag / q;
        if (2 * (mag % q) >= q) qt += 1;
        *out = (long long)(x < 0 ? -qt : qt);
        *ovf = 0;
    }
}

static bool run_case(long long n, long long shift, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<long long> values(n), out(n, 7), eout(n);
    std::vector<signed char> ovf(n, 7), eovf(n);
    for (long long i = 0; i < n; ++i) {
        switch (rng() & 3) {
            case 0: values[i] = (long long)(rng() % 2000000) - 1000000; break;
            case 1: values[i] = (long long)(rng() % (1ull << 41)) - (1ll << 40); break;
            default: values[i] = (long long)rng(); break;
        }
    }
    if (n > 0) values[0] = (-0x7FFFFFFFFFFFFFFFll) - 1;
    if (n > 1) values[1] = 0x7FFFFFFFFFFFFFFFll;
    if (n > 2) values[2] = -922337203685477580ll;
    for (long long i = 0; i < n; ++i) reference(values[i], shift, &eout[i], &eovf[i]);
    long long* d_values = upload(values); long long* d_out = upload(out); signed char* d_ovf = upload(ovf);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_out, d_ovf, n, shift);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(ovf.data(), d_ovf, n, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i)
        if (out[i] != eout[i] || ovf[i] != eovf[i]) { ok = false; fprintf(stderr, "n=%lld shift=%lld row %lld: %lld/%d != %lld/%d\n", n, shift, i, out[i], ovf[i], eout[i], eovf[i]); }
    cudaFree(d_values); cudaFree(d_out); cudaFree(d_ovf);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long shifts[] = {-18, -5, -1, 0, 1, 3, 18};
    bool ok = true;
    for (long long n : sizes) for (long long s : shifts) ok = run_case(n, s, (unsigned)(n + s + 100)) && ok;
    printf("decimal_rescale cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
