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
    std::vector<long long> values(n);
    for (long long i = 0; i < n; ++i) {
        switch (variant) {
            case 0: values[i] = (long long)rng(); break;
            case 1: values[i] = (long long)(rng() % 6) - 3; break;
            case 2: values[i] = (n - 1 - i) - n; break;
            default: values[i] = (-0x7FFFFFFFFFFFFFFFll) - 1; break;
        }
    }
    long long expect = -1;
    for (long long i = 0; i < n; ++i) if (expect < 0 || values[i] < values[expect]) expect = i;
    std::vector<int> row(1, -1);
    long long* d_values = upload(values); int* d_row = upload(row);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_row, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(row.data(), d_row, sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = row[0] == expect;
    if (!ok) fprintf(stderr, "n=%lld variant=%d: %d != %lld\n", n, variant, row[0], expect);
    cudaFree(d_values); cudaFree(d_row);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 4; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("argmin cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
