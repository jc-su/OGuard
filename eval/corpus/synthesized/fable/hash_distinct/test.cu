// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>
#include "kernel.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

template <class T> static T* upload(const std::vector<T>& v) {
    T* d; CHECK(cudaMalloc(&d, (v.size() + 1) * sizeof(T)));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static const int EMPTY_ROW = 0x7FFFFFFF;

static bool run_case(long long n, int variant, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) {
        switch (variant) {
            case 0: keys[i] = (long long)rng(); break;
            case 1: keys[i] = (long long)(rng() % std::max(1LL, n / 50)); break;
            default: keys[i] = i < n / 2 ? i * 4096 : 12345; break;
        }
        if (keys[i] == -1) keys[i] = -2;
    }
    long long capacity = 1;
    while (capacity < 2 * n) capacity *= 2;
    std::vector<long long> table_keys(capacity, -1);
    std::vector<int> table_rows(capacity, EMPTY_ROW);
    long long* d_keys = upload(keys); long long* d_tk = upload(table_keys); int* d_tr = upload(table_rows);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_keys, d_tk, d_tr, n, capacity);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(table_keys.data(), d_tk, capacity * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(table_rows.data(), d_tr, capacity * sizeof(int), cudaMemcpyDeviceToHost));
    std::unordered_map<long long, long long> first;
    for (long long i = 0; i < n; ++i) if (!first.count(keys[i])) first[keys[i]] = i;
    bool ok = true;
    long long occupied = 0;
    std::unordered_map<long long, int> seen;
    for (long long s = 0; s < capacity && ok; ++s) {
        long long k = table_keys[s];
        if (k == -1) { if (table_rows[s] != EMPTY_ROW) { ok = false; fprintf(stderr, "empty slot %lld has row\n", s); } continue; }
        ++occupied;
        auto it = first.find(k);
        if (it == first.end()) { ok = false; fprintf(stderr, "slot %lld holds unknown key\n", s); break; }
        if (table_rows[s] != it->second) { ok = false; fprintf(stderr, "slot %lld row %d != %lld\n", s, table_rows[s], it->second); break; }
        if (seen[k]++) { ok = false; fprintf(stderr, "key %lld twice\n", k); break; }
        long long start = (long long)((((unsigned long long)k) * 0x9E3779B97F4A7C15ull) >> 32) & (capacity - 1);
        for (long long q = start; q != s; q = (q + 1) & (capacity - 1))
            if (table_keys[q] == -1) { ok = false; fprintf(stderr, "key %lld not reachable\n", k); break; }
    }
    if (ok && occupied != (long long)first.size()) { ok = false; fprintf(stderr, "occupied %lld != %zu\n", occupied, first.size()); }
    if (!ok) fprintf(stderr, "n=%lld variant=%d failed\n", n, variant);
    cudaFree(d_keys); cudaFree(d_tk); cudaFree(d_tr);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) for (int v = 0; v < 3; ++v) ok = run_case(n, v, (unsigned)(n + v)) && ok;
    printf("hash_distinct cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
