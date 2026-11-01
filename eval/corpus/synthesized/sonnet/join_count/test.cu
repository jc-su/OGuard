#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "kernel.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)
template <class T> static T* up(const std::vector<T>& v) {
    T* d; if (cudaMalloc(&d, v.size() * sizeof(T) + 1) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}
template <class T> static T* zeros(size_t n) {
    T* d; if (cudaMalloc(&d, n * sizeof(T) + 1) != cudaSuccess) return nullptr;
    cudaMemset(d, 0, n * sizeof(T)); return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n); cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return v;
}
static long long blocks_for(long long n) { return std::max(1LL, (n + 255) / 256); }

int main() {
    std::mt19937_64 rng(1);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (long long b : {0LL, 1LL, n}) {
        std::vector<long long> build(b), probe(n);
        for (auto& v : build) v = (long long)(rng() % (110 + n / 100)) - 50;
        for (auto& v : probe) v = (long long)(rng() % (120 + n / 100)) - 60;
        std::sort(build.begin(), build.end());
        std::vector<int> ref(n);
        for (long long i = 0; i < n; i++)
            ref[i] = (int)(std::upper_bound(build.begin(), build.end(), probe[i]) - std::lower_bound(build.begin(), build.end(), probe[i]));
        long long* d_b = up(build); long long* d_p = up(probe); int* d_c = zeros<int>(n);
        kernel<<<blocks_for(n), 256>>>(d_b, d_p, d_c, b, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_c, n) != ref) { printf("FAIL n=%lld b=%lld\n", n, b); return 1; }
        cudaFree(d_b); cudaFree(d_p); cudaFree(d_c);
    }
    printf("PASS\n");
    return 0;
}
