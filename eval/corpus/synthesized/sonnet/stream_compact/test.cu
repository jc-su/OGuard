#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "kernel.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    std::mt19937_64 rng(1);
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        std::vector<long long> values(n), ref;
        std::vector<signed char> keep(n);
        for (long long i = 0; i < n; i++) {
            values[i] = (long long)rng();
            keep[i] = (rng() % 10 < 3) ? (signed char)(rng() % 255 - 127) : 0;
            if (keep[i]) ref.push_back(values[i]);
        }
        long long *d_values, *d_out; signed char* d_keep; int* d_count;
        CK(cudaMalloc(&d_values, n * 8)); CK(cudaMalloc(&d_out, n * 8));
        CK(cudaMalloc(&d_keep, n)); CK(cudaMalloc(&d_count, 4));
        CK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_keep, keep.data(), n, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_count, 0, 4));
        long long blocks = std::max(1LL, (n + 255) / 256);
        kernel<<<blocks, 256>>>(d_values, d_keep, d_out, d_count, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int count = -1;
        CK(cudaMemcpy(&count, d_count, 4, cudaMemcpyDeviceToHost));
        if (count != (int)ref.size()) { printf("FAIL n=%lld count %d vs %zu\n", n, count, ref.size()); return 1; }
        std::vector<long long> got(count);
        CK(cudaMemcpy(got.data(), d_out, count * 8, cudaMemcpyDeviceToHost));
        std::sort(got.begin(), got.end()); std::sort(ref.begin(), ref.end());
        if (got != ref) { printf("FAIL n=%lld values\n", n); return 1; }
        printf("n %lld ok\n", n);
        cudaFree(d_values); cudaFree(d_out); cudaFree(d_keep); cudaFree(d_count);
    }
    printf("PASS\n");
    return 0;
}
