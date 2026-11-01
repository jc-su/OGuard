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
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) for (long long w : {1LL, 2LL, 3LL, 63LL, 64LL, 65LL, 100LL, 256LL, 257LL, 1000LL, 1024LL}) {
        std::vector<long long> v(n), ref(n);
        for (auto& x : v) x = (long long)rng();
        // monotone deque reference
        std::vector<long long> dq(n); long long head = 0, tail = 0;
        for (long long i = 0; i < n; i++) {
            while (tail > head && v[dq[tail - 1]] <= v[i]) tail--;
            dq[tail++] = i;
            if (dq[head] <= i - w) head++;
            ref[i] = v[dq[head]];
        }
        long long* d_v = up(v); long long* d_o = zeros<long long>(n);
        long long blocks = w <= 64 ? blocks_for(n) : std::max(1LL, (n + w - 1) / w);
        kernel<<<blocks, 256>>>(d_v, d_o, n, w);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_o, n) != ref) { printf("FAIL n=%lld w=%lld\n", n, w); return 1; }
        cudaFree(d_v); cudaFree(d_o);
    }
    printf("PASS\n");
    return 0;
}
