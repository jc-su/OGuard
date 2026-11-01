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

static bool run_case(long long b, long long p, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long span = std::max(1LL, std::max(b, p) / 4);
    std::vector<long long> build(b), probe(p);
    for (auto& x : build) x = (long long)(rng() % span);
    std::sort(build.begin(), build.end());
    for (auto& x : probe) x = (long long)(rng() % (span + 4)) - 2;
    if (b > 0 && p > 0) for (long long i = 0; i < p / 2; ++i) probe[i] = build[rng() % b];
    std::vector<int> offsets(p + 1, 0), first(p);
    for (long long i = 0; i < p; ++i) {
        auto lo = std::lower_bound(build.begin(), build.end(), probe[i]);
        auto hi = std::upper_bound(build.begin(), build.end(), probe[i]);
        first[i] = (int)(lo - build.begin());
        offsets[i + 1] = offsets[i] + (int)(hi - lo);
    }
    long long total = offsets[p];
    std::vector<int> ob(total, 7), op(total, 7), eb(total), ep(total);
    for (long long i = 0; i < p; ++i)
        for (int q = offsets[i], j = 0; q < offsets[i + 1]; ++q, ++j) { eb[q] = first[i] + j; ep[q] = (int)i; }
    long long* d_build = upload(build); long long* d_probe = upload(probe); int* d_offsets = upload(offsets);
    int* d_ob = upload(ob); int* d_op = upload(op);
    long long blocks = (total + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_build, d_probe, d_offsets, d_ob, d_op, b, p);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(ob.data(), d_ob, total * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(op.data(), d_op, total * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long q = 0; q < total && ok; ++q)
        if (ob[q] != eb[q] || op[q] != ep[q]) { ok = false; fprintf(stderr, "b=%lld p=%lld pair %lld: (%d,%d) != (%d,%d)\n", b, p, q, ob[q], op[q], eb[q], ep[q]); }
    cudaFree(d_build); cudaFree(d_probe); cudaFree(d_offsets); cudaFree(d_ob); cudaFree(d_op);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long p : sizes) for (long long b : sizes) ok = run_case(b, p, (unsigned)(p + b)) && ok;
    printf("join_materialize cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
