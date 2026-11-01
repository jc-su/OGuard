// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "kernel.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static bool run_case(long long n, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long nbytes = (n + 7) / 8;
    long long lo = (long long)(rng() >> 2) - (1LL << 61);
    long long hi = lo + (long long)(rng() >> 3);
    std::vector<long long> values(n);
    std::vector<unsigned char> valid(nbytes), out(nbytes, 0xA5), expect(nbytes, 0);
    for (long long i = 0; i < n; ++i) {
        switch (rng() & 3) {
            case 0: values[i] = lo; break;
            case 1: values[i] = hi; break;
            case 2: values[i] = lo + (long long)(rng() % (unsigned long long)(hi - lo + 1)); break;
            default: values[i] = (long long)rng(); break;
        }
    }
    for (long long b = 0; b < nbytes; ++b) valid[b] = (unsigned char)rng();
    for (long long i = 0; i < n; ++i) {
        bool v = (valid[i / 8] >> (i % 8)) & 1;
        if (v && values[i] >= lo && values[i] <= hi) expect[i / 8] |= (unsigned char)(1u << (i % 8));
    }
    long long *d_values; unsigned char *d_valid, *d_out;
    CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(long long)));
    CHECK(cudaMalloc(&d_valid, nbytes + 1));
    CHECK(cudaMalloc(&d_out, nbytes + 1));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_valid, valid.data(), nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out, out.data(), nbytes, cudaMemcpyHostToDevice));
    long long blocks = (nbytes + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_valid, d_out, n, lo, hi);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long b = 0; b < nbytes && ok; ++b) if (out[b] != expect[b]) { ok = false; fprintf(stderr, "n=%lld byte %lld: %u != %u\n", n, b, out[b], expect[b]); }
    cudaFree(d_values); cudaFree(d_valid); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("between_bitmap cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
