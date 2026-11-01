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
    long long run = 0;
    for (long long i = 0; i < n; ++i) {
        switch (variant) {
            case 0: if (i == 0 || (rng() % 10) < 3) ++run; values[i] = run * 7 - 3; break;
            case 1: values[i] = -5; break;
            case 2: values[i] = i; break;
            default: values[i] = (i / 50001) % 3; break;
        }
    }
    std::vector<int> ends0(n, -7), ends(n), e_ends;
    std::vector<long long> vals0(n, -9), vals(n), e_vals;
    for (long long i = 0; i < n; ++i)
        if (i == n - 1 || values[i] != values[i + 1]) { e_ends.push_back((int)(i + 1)); e_vals.push_back(values[i]); }
    long long r = (long long)e_ends.size();
    std::vector<int> count(1, 0);
    long long* d_values = upload(values); int* d_ends = upload(ends0); long long* d_vals = upload(vals0); int* d_count = upload(count);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_values, d_ends, d_vals, d_count, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(ends.data(), d_ends, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vals.data(), d_vals, n * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(count.data(), d_count, sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = count[0] == r;
    if (!ok) fprintf(stderr, "n=%lld variant=%d: count %d != %lld\n", n, variant, count[0], r);
    for (long long j = 0; j < n && ok; ++j) {
        int ee = j < r ? e_ends[j] : ends0[j];
        long long ev = j < r ? e_vals[j] : vals0[j];
        if (ends[j] != ee || vals[j] != ev) { ok = false; fprintf(stderr, "n=%lld variant=%d: entry %lld (%d,%lld) != (%d,%lld)\n", n, variant, j, ends[j], vals[j], ee, ev); }
    }
    cudaFree(d_values); cudaFree(d_ends); cudaFree(d_vals); cudaFree(d_count);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 4; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("run_end_encode cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
