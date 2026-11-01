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
    for (long long n : {0LL, 1LL, 7LL, 8LL, 9LL, 1000LL, 1000003LL}) for (int pct : {0, 50, 100}) {
        std::vector<unsigned char> valid((n + 7) / 8, 0);
        long long ref = 0;
        for (long long i = 0; i < n; i++) {
            if ((int)(rng() % 100) < pct) valid[i / 8] |= 1 << (i % 8); else ref++;
        }
        if (n % 8) valid.back() |= (unsigned char)(0xFF << (n % 8));  // garbage past n
        unsigned char* d_v = up(valid); long long* d_c = zeros<long long>(1);
        long long blocks = std::min(1024LL, blocks_for((n + 7) / 8));
        kernel<<<blocks, 256>>>(d_v, d_c, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long got = down(d_c, 1)[0];
        if (got != ref) { printf("FAIL n=%lld pct=%d %lld vs %lld\n", n, pct, got, ref); return 1; }
        cudaFree(d_v); cudaFree(d_c);
    }
    printf("PASS\n");
    return 0;
}
