// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <algorithm>
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

static bool run_case(long long n, long long r, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long span = std::max(1LL, 4 * std::max(n, r));
    std::vector<long long> left(n), right(r);
    for (auto& x : left) x = (long long)(rng() % (2 * span)) - span;
    for (auto& x : right) x = (long long)(rng() % (2 * span)) - span;
    std::sort(right.begin(), right.end());
    if (n > 0 && r > 0) for (long long i = 0; i < n / 4; ++i) left[i] = right[rng() % r];
    long long tolerance = (long long)(rng() % 8);
    std::vector<int> out(n, 7), expect(n);
    for (long long i = 0; i < n; ++i) {
        long long j = (long long)(std::upper_bound(right.begin(), right.end(), left[i]) - right.begin()) - 1;
        expect[i] = (j >= 0 && left[i] - right[j] <= tolerance) ? (int)j : -1;
    }
    long long* d_left = upload(left); long long* d_right = upload(right); int* d_out = upload(out);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_left, d_right, d_out, n, r, tolerance);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld r=%lld row %lld: %d != %d\n", n, r, i, out[i], expect[i]); }
    cudaFree(d_left); cudaFree(d_right); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (long long r : sizes) ok = run_case(n, r, (unsigned)(n + r)) && ok;
    printf("asof_join cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
