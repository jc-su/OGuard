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

static bool run_case(long long n, long long g, unsigned seed) {
    std::mt19937_64 rng(seed);
    long long nbytes = (n + 7) / 8;
    std::vector<int> group(n);
    std::vector<long long> value(n), counts(g, 0), sums(g, 0), ec(g, 0);
    std::vector<unsigned long long> es(g, 0);
    std::vector<unsigned char> valid(nbytes);
    for (long long i = 0; i < n; ++i) { group[i] = (int)(rng() % g); value[i] = (long long)rng(); }
    for (auto& b : valid) b = (unsigned char)rng();
    for (long long i = 0; i < n; ++i)
        if ((valid[i / 8] >> (i % 8)) & 1) { ec[group[i]] += 1; es[group[i]] += (unsigned long long)value[i]; }
    int* d_group = upload(group); long long* d_value = upload(value); unsigned char* d_valid = upload(valid);
    long long* d_counts = upload(counts); long long* d_sums = upload(sums);
    long long blocks = (n + TILE - 1) / TILE; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_group, d_value, d_valid, d_counts, d_sums, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(counts.data(), d_counts, g * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sums.data(), d_sums, g * sizeof(long long), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long k = 0; k < g && ok; ++k)
        if (counts[k] != ec[k] || (unsigned long long)sums[k] != es[k]) { ok = false; fprintf(stderr, "n=%lld g=%lld group %lld differs\n", n, g, k); }
    cudaFree(d_group); cudaFree(d_value); cudaFree(d_valid); cudaFree(d_counts); cudaFree(d_sums);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    long long groups[] = {1, 7, 1024};
    bool ok = true;
    for (long long n : sizes) for (long long g : groups) ok = run_case(n, g, (unsigned)(n + g)) && ok;
    printf("group_sum_valid cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
