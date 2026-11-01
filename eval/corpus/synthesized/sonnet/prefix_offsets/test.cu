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
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (int hi : {3, 2000}) {
        std::vector<int> lengths(n), ref(n + 1, 0);
        for (long long i = 0; i < n; i++) { lengths[i] = (int)(rng() % (hi + 1)); ref[i + 1] = ref[i] + lengths[i]; }
        int* d_l = up(lengths); int* d_o = zeros<int>(n + 1);
        cudaMemset(d_o, 0xFB, (n + 1) * 4);
        kernel<<<blocks_for(n), 256>>>(d_l, d_o, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_o, n + 1) != ref) { printf("FAIL n=%lld hi=%d\n", n, hi); return 1; }
        cudaFree(d_l); cudaFree(d_o);
    }
    printf("PASS\n");
    return 0;
}
