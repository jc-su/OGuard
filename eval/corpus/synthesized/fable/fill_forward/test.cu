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

static bool run_case(long long n, int variant, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long nbytes = (n + 7) / 8;
    std::vector<long long> values(n), out(n, 7), expect(n);
    std::vector<unsigned char> valid(nbytes, 0);
    for (long long i = 0; i < n; ++i) {
        values[i] = (long long)rng();
        bool bit;
        switch (variant) {
            case 0: bit = rng() & 1; break;
            case 1: bit = false; break;
            case 2: bit = true; break;
            case 3: bit = i == 0; break;
            default: bit = (i % 50001) == 3; break;
        }
        if (bit) valid[i / 8] |= (unsigned char)(1u << (i % 8));
    }
    if (n % 8 && nbytes) valid[nbytes - 1] |= (unsigned char)((0xFFu << (n % 8)) & 0xFFu);  // junk padding bits
    long long last = -1;
    for (long long i = 0; i < n; ++i) {
        if ((valid[i / 8] >> (i % 8)) & 1) last = i;
        expect[i] = last >= 0 ? values[last] : 0;
    }
    long long* d_values = upload(values); unsigned char* d_valid = upload(valid); long long* d_out = upload(out);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_valid, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld variant=%d row %lld: %lld != %lld\n", n, variant, i, out[i], expect[i]); }
    cudaFree(d_values); cudaFree(d_valid); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 5; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("fill_forward cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
