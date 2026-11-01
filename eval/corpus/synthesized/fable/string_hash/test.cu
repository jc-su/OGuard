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

static bool run_case(long long n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> offsets(n + 1, 0);
    for (long long i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + (int)(rng() % 24);
    if (n > 100) for (int k = 0; k < 5; ++k) {
        long long r = rng() % n;
        for (long long i = r + 1; i <= n; ++i) offsets[i] += 3000;
    }
    std::vector<unsigned char> chars(offsets[n]);
    for (auto& c : chars) c = (unsigned char)rng();
    std::vector<unsigned int> out(n, 7), expect(n);
    for (long long i = 0; i < n; ++i) {
        unsigned int h = 2166136261u;
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) h = (h ^ chars[p]) * 16777619u;
        expect[i] = h;
    }
    int* d_offsets = upload(offsets); unsigned char* d_chars = upload(chars); unsigned int* d_out = upload(out);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_offsets, d_chars, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld row %lld: %u != %u\n", n, i, out[i], expect[i]); }
    cudaFree(d_offsets); cudaFree(d_chars); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("string_hash cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
