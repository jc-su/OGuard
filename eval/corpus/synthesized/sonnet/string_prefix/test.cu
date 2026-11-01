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
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (long long k : {1LL, 7LL, 64LL}) {
        std::vector<int> offsets(n + 1, 0);
        for (long long i = 0; i < n; i++) offsets[i + 1] = offsets[i] + (int)(rng() % 90);
        std::vector<unsigned char> chars(offsets[n]);
        for (auto& c : chars) c = 1 + rng() % 255;
        std::vector<unsigned char> ref(n * k, 0);
        for (long long i = 0; i < n; i++)
            for (long long j = 0; j < k && j < offsets[i + 1] - offsets[i]; j++) ref[i * k + j] = chars[offsets[i] + j];
        int* d_off = up(offsets); unsigned char* d_chars = up(chars); unsigned char* d_out = zeros<unsigned char>(n * k);
        cudaMemset(d_out, 77, n * k);
        kernel<<<blocks_for(n * k), 256>>>(d_off, d_chars, d_out, n, k);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_out, n * k) != ref) { printf("FAIL n=%lld k=%lld\n", n, k); return 1; }
        cudaFree(d_off); cudaFree(d_chars); cudaFree(d_out);
    }
    printf("PASS\n");
    return 0;
}
