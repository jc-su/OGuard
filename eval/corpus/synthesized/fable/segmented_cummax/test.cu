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
    std::vector<long long> values(n), out(n, 7), expect(n);
    std::vector<int> first(n);
    for (long long i = 0; i < n; ++i) {
        bool start;
        switch (variant) {
            case 0: values[i] = (long long)rng(); start = (rng() % 10) < 3; break;
            case 1: values[i] = (long long)rng(); start = false; break;
            case 2: values[i] = (long long)rng(); start = true; break;
            default: values[i] = i - (1ll << 62); start = (i % 50001) == 0; break;
        }
        if (i == 0) start = true;
        first[i] = start ? (int)i : first[i - 1];
    }
    for (long long i = 0; i < n; ++i) expect[i] = (first[i] == i || values[i] > expect[i - 1]) ? values[i] : expect[i - 1];
    long long* d_values = upload(values); int* d_first = upload(first); long long* d_out = upload(out);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_first, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld variant=%d row %lld: %lld != %lld\n", n, variant, i, out[i], expect[i]); }
    cudaFree(d_values); cudaFree(d_first); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 4; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("segmented_cummax cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
