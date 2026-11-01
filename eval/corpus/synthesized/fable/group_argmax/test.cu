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

static bool run_case(long long n, long long g, int variant, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> group(n), best(g, -1), expect(g, -1);
    std::vector<long long> value(n);
    for (long long i = 0; i < n; ++i) {
        group[i] = (int)(rng() % g);
        switch (variant) {
            case 0: value[i] = (long long)rng(); break;
            case 1: value[i] = (long long)(rng() % 4) - 2; break;
            default: value[i] = i; break;
        }
    }
    for (long long i = 0; i < n; ++i) {
        int& e = expect[group[i]];
        if (e < 0 || value[i] > value[e]) e = (int)i;
    }
    int* d_group = upload(group); long long* d_value = upload(value); int* d_best = upload(best);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_group, d_value, d_best, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(best.data(), d_best, g * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long k = 0; k < g && ok; ++k) if (best[k] != expect[k]) { ok = false; fprintf(stderr, "n=%lld g=%lld variant=%d group %lld: %d != %d\n", n, g, variant, k, best[k], expect[k]); }
    cudaFree(d_group); cudaFree(d_value); cudaFree(d_best);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long groups[] = {1, 7, 1000, 1 << 20};
    bool ok = true;
    for (long long n : sizes) for (long long g : groups) for (int v = 0; v < 3; ++v) ok = run_case(n, g, v, (unsigned)(n + g + v)) && ok;
    printf("group_argmax cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
