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

// CPU reference: walk months from the day number using days_from_civil.
static long long days_from_civil(long long y, int m, int d) {
    y -= m <= 2;
    long long era = (y >= 0 ? y : y - 399) / 400;
    long long yoe = y - era * 400;
    long long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}
static int ref_month_start(int day) {
    long long z = (long long)day + 719468;
    long long era = (z >= 0 ? z : z - 146096) / 146097;
    long long doe = z - era * 146097;
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long y = yoe + era * 400;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long long mp = (5 * doy + 2) / 153;
    int m = (int)(mp < 10 ? mp + 3 : mp - 9);
    return (int)days_from_civil(y + (m <= 2), m, 1);
}

int main() {
    std::mt19937_64 rng(1);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL, 3652059LL}) {
        std::vector<int> days(n), ref(n);
        for (long long i = 0; i < n; i++) days[i] = (n == 3652059LL) ? (int)(i - 719162) : (int)(rng() % 3652059) - 719162;
        for (long long i = 0; i < n; i++) ref[i] = ref_month_start(days[i]);
        int* d_d = up(days); int* d_o = zeros<int>(n);
        kernel<<<blocks_for(n), 256>>>(d_d, d_o, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        if (down(d_o, n) != ref) { printf("FAIL n=%lld\n", n); return 1; }
        cudaFree(d_d); cudaFree(d_o);
    }
    printf("PASS\n");
    return 0;
}
