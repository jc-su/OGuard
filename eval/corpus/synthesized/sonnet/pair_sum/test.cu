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
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (int c = 0; c < 3; c++) {
        long long l = c == 0 ? 1 : c == 1 ? 1000 : std::max(1LL, n);
        long long r = c == 0 ? 1 : c == 1 ? 37 : std::max(1LL, n / 3);
        std::vector<int> li(n), ri(n);
        std::vector<double> left(l), right(r), ref(n);
        for (auto& x : left) x = dist(rng);
        for (auto& x : right) x = dist(rng);
        for (long long i = 0; i < n; i++) {
            li[i] = (int)(rng() % l); ri[i] = (int)(rng() % r);
            ref[i] = left[li[i]] + right[ri[i]];
        }
        int* d_li = up(li); int* d_ri = up(ri); double* d_l = up(left); double* d_r = up(right); double* d_o = zeros<double>(n);
        kernel<<<blocks_for(n), 256>>>(d_li, d_ri, d_l, d_r, d_o, n, l, r);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_o, n) != ref) { printf("FAIL n=%lld c=%d\n", n, c); return 1; }
        cudaFree(d_li); cudaFree(d_ri); cudaFree(d_l); cudaFree(d_r); cudaFree(d_o);
    }
    printf("PASS\n");
    return 0;
}
