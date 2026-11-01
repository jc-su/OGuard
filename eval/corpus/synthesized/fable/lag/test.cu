// Test the lag kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o lag_test kernel.cu test.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* values,
                                  const int* first,
                                  double* out,
                                  long long n,
                                  long long k);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err));                              \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

// Partition starts: mostly partitions of 1 to 40 rows, with a few of up to
// 3000 rows so that k = 1024 has in-partition lags.
static std::vector<int> make_first(long long n, std::mt19937_64& rng) {
    std::vector<int> first(n);
    long long start = 0;
    while (start < n) {
        long long size = (rng() % 100 < 95) ? 1 + (long long)(rng() % 40) : 1000 + (long long)(rng() % 2001);
        for (long long i = start; i < start + size && i < n; ++i) first[i] = (int)start;
        start += size;
    }
    return first;
}

static bool check(long long n, long long k, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> values(n);
    for (long long i = 0; i < n; ++i) values[i] = dist(rng);
    std::vector<int> first = make_first(n, rng);
    std::vector<double> expected(n);
    long long lagged = 0;
    for (long long i = 0; i < n; ++i) {
        long long j = i - k;
        bool inside = j >= first[i];
        expected[i] = inside ? values[j] : nan("");
        lagged += inside;
    }

    double *d_values, *d_out;
    int* d_first;
    CUDA_CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_first, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first, first.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, (n + 1) * sizeof(double)));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_values, d_first, d_out, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) {
        ok = std::isnan(expected[i]) ? std::isnan(out[i]) : out[i] == expected[i];
    }

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("n=%lld k=%lld: %s (%lld lagged rows)\n", n, k, ok ? "PASS" : "FAIL", lagged);
    return ok;
}

int main() {
    std::mt19937_64 rng(5);
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long lags[] = {1, 3, 1024};
    int failures = 0;
    for (long long n : sizes) {
        for (long long k : lags) {
            if (!check(n, k, rng)) ++failures;
        }
    }
    return failures ? 1 : 0;
}
