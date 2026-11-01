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
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (long long p : {1LL, 7LL, 256LL, 1000LL, 1024LL}) {
        std::vector<long long> keys(n);
        std::vector<int> ref(n), refc(p, 0);
        for (long long i = 0; i < n; i++) {
            keys[i] = (i < n / 2 && n > 10) ? (long long)(rng() % 10) - 5 : (long long)rng();
            unsigned long long h = ((unsigned long long)keys[i] * 0x9E3779B97F4A7C15ULL) >> 32;
            ref[i] = (int)(h % (unsigned long long)p);
            refc[ref[i]]++;
        }
        long long* d_k = up(keys); int* d_p = zeros<int>(n); int* d_c = zeros<int>(p);
        kernel<<<std::min(1024LL, blocks_for(n)), 256>>>(d_k, d_p, d_c, n, p);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_p, n) != ref || down(d_c, p) != refc) { printf("FAIL n=%lld p=%lld\n", n, p); return 1; }
        cudaFree(d_k); cudaFree(d_p); cudaFree(d_c);
    }
    printf("PASS\n");
    return 0;
}
