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

static bool run_case(long long n, long long offset, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long src_bytes = (offset + n + 7) / 8, nbytes = (n + 7) / 8;
    std::vector<unsigned char> source(src_bytes), out(nbytes, 0xA5), expect(nbytes, 0);
    for (auto& c : source) c = (unsigned char)rng();
    for (long long i = 0; i < n; ++i) {
        long long s = offset + i;
        if ((source[s / 8] >> (s % 8)) & 1) expect[i / 8] |= (unsigned char)(1u << (i % 8));
    }
    unsigned char* d_source = upload(source); unsigned char* d_out = upload(out);
    long long blocks = (nbytes + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_source, d_out, n, offset);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long b = 0; b < nbytes && ok; ++b) if (out[b] != expect[b]) { ok = false; fprintf(stderr, "n=%lld offset=%lld byte %lld: %u != %u\n", n, offset, b, out[b], expect[b]); }
    cudaFree(d_source); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long offsets[] = {0, 3, 8, 13, 1000001};
    bool ok = true;
    for (long long n : sizes) for (long long o : offsets) ok = run_case(n, o, (unsigned)(n + o)) && ok;
    printf("bitmap_slice cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
