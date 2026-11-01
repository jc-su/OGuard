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
        int extra = 5000;
        for (long long i = r + 1; i <= n; ++i) offsets[i] += extra;
    }
    std::vector<unsigned char> chars(offsets[n]);
    for (auto& c : chars) c = (unsigned char)rng();
    long long m = n < 10 ? n : n + 7;
    std::vector<int> idx(m), out_offsets(m + 1, 0);
    for (long long i = 0; i < m; ++i) idx[i] = (int)(rng() % n);
    for (long long i = 0; i < m; ++i) out_offsets[i + 1] = out_offsets[i] + offsets[idx[i] + 1] - offsets[idx[i]];
    long long total = out_offsets[m];
    std::vector<unsigned char> expect(total), out(total, 0xA5);
    for (long long i = 0; i < m; ++i)
        for (int k = offsets[idx[i]], o = out_offsets[i]; k < offsets[idx[i] + 1]; ++k, ++o) expect[o] = chars[k];

    int* d_offsets = upload(offsets); unsigned char* d_chars = upload(chars);
    int* d_idx = upload(idx); int* d_out_offsets = upload(out_offsets); unsigned char* d_out = upload(out);
    long long blocks = (total + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_offsets, d_chars, d_idx, d_out_offsets, d_out, n, m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, total, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long p = 0; p < total && ok; ++p) if (out[p] != expect[p]) { ok = false; fprintf(stderr, "n=%lld byte %lld: %u != %u\n", n, p, out[p], expect[p]); }
    cudaFree(d_offsets); cudaFree(d_chars); cudaFree(d_idx); cudaFree(d_out_offsets); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("gather_strings cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
