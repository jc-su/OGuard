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
    long long shapes[][2] = {{0, 5}, {5, 0}, {0, 0}, {1, 1}, {1, 1000}, {1000, 1}, {1000003, 1}, {1, 1000003},
                             {31, 33}, {1000, 1003}, {1003, 997}, {2, 500001}};
    for (auto& s : shapes) {
        long long rows = s[0], cols = s[1], n = rows * cols;
        std::vector<int> src(n), ref(n);
        for (long long i = 0; i < rows; i++) for (long long j = 0; j < cols; j++) {
            src[i * cols + j] = (int)rng();
            ref[j * rows + i] = src[i * cols + j];
        }
        int* d_s = up(src); int* d_o = zeros<int>(n);
        long long ntiles = ((rows + 31) / 32) * ((cols + 31) / 32);
        kernel<<<std::max(1LL, ntiles), 256>>>(d_s, d_o, rows, cols);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_o, n) != ref) { printf("FAIL %lld x %lld\n", rows, cols); return 1; }
        cudaFree(d_s); cudaFree(d_o);
    }
    printf("PASS\n");
    return 0;
}
