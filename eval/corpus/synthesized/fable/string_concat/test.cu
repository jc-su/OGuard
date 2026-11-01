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

static void column(long long n, std::mt19937_64& rng, std::vector<int>& offsets, std::vector<unsigned char>& chars) {
    offsets.assign(n + 1, 0);
    for (long long i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + (int)(rng() % 16);
    if (n > 100) for (int k = 0; k < 5; ++k) {
        long long r = rng() % n;
        for (long long i = r + 1; i <= n; ++i) offsets[i] += 4000;
    }
    chars.resize(offsets[n]);
    for (auto& c : chars) c = (unsigned char)rng();
}

static bool run_case(long long n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> a_offsets, b_offsets, out_offsets(n + 1);
    std::vector<unsigned char> a_chars, b_chars;
    column(n, rng, a_offsets, a_chars);
    column(n, rng, b_offsets, b_chars);
    for (long long i = 0; i <= n; ++i) out_offsets[i] = a_offsets[i] + b_offsets[i];
    long long total = out_offsets[n];
    std::vector<unsigned char> out(total, 0xA5), expect(total);
    for (long long i = 0; i < n; ++i) {
        long long o = out_offsets[i];
        for (int k = a_offsets[i]; k < a_offsets[i + 1]; ++k) expect[o++] = a_chars[k];
        for (int k = b_offsets[i]; k < b_offsets[i + 1]; ++k) expect[o++] = b_chars[k];
    }
    int* d_ao = upload(a_offsets); unsigned char* d_ac = upload(a_chars);
    int* d_bo = upload(b_offsets); unsigned char* d_bc = upload(b_chars);
    int* d_oo = upload(out_offsets); unsigned char* d_out = upload(out);
    long long blocks = (total + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_ao, d_ac, d_bo, d_bc, d_oo, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, total, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long p = 0; p < total && ok; ++p) if (out[p] != expect[p]) { ok = false; fprintf(stderr, "n=%lld byte %lld: %u != %u\n", n, p, out[p], expect[p]); }
    cudaFree(d_ao); cudaFree(d_ac); cudaFree(d_bo); cudaFree(d_bc); cudaFree(d_oo); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("string_concat cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
