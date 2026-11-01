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

static bool run_case(long long n, long long bits, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long w = (n * bits + 31) / 32;
    std::vector<unsigned int> packed(w), out(n, 7), expect(n);
    for (auto& x : packed) x = (unsigned int)rng();
    for (long long i = 0; i < n; ++i) {
        unsigned int code = 0;
        for (long long k = 0; k < bits; ++k) {
            long long p = i * bits + k;
            code |= ((packed[p / 32] >> (p % 32)) & 1u) << k;
        }
        expect[i] = code;
    }
    unsigned int* d_packed = upload(packed); unsigned int* d_out = upload(out);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_packed, d_out, n, w, bits);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld bits=%lld code %lld: %u != %u\n", n, bits, i, out[i], expect[i]); }
    cudaFree(d_packed); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long widths[] = {1, 3, 7, 8, 13, 31, 32};
    bool ok = true;
    for (long long n : sizes) for (long long b : widths) ok = run_case(n, b, (unsigned)(n + b)) && ok;
    printf("bit_unpack cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
