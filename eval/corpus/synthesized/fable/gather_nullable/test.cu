// Test the gather_nullable kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o gather_nullable_test kernel.cu test.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* col,
                                  const int* idx,
                                  double* out,
                                  long long m,
                                  long long n);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err));                              \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static bool check(long long m, long long n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> col(n);
    for (long long i = 0; i < n; ++i) col[i] = dist(rng);
    std::vector<int> idx(m);
    for (long long i = 0; i < m; ++i) {
        if (n == 0 || (rng() % 10) < 3) {
            idx[i] = -1;
        } else {
            idx[i] = (int)(rng() % (unsigned long long)n);
        }
    }
    std::vector<double> expected(m);
    for (long long i = 0; i < m; ++i) expected[i] = idx[i] < 0 ? nan("") : col[idx[i]];

    double *d_col, *d_out;
    int* d_idx;
    CUDA_CHECK(cudaMalloc(&d_col, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_idx, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (m + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_col, col.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, idx.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, (m + 1) * sizeof(double)));

    unsigned blocks = (unsigned)((m + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_col, d_idx, d_out, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(m);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, m * sizeof(double), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < m && ok; ++i) {
        if (std::isnan(expected[i])) {
            ok = std::isnan(out[i]);
        } else {
            ok = out[i] == expected[i];
        }
    }

    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("m=%lld n=%lld: %s\n", m, n, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937_64 rng(2);
    const long long cases[][2] = {{0, 0},       {0, 1000},         {1, 0},          {1, 1},
                                  {1000, 1},    {1000, 1000},      {1000003, 1000003}, {1000003, 1000}};
    int failures = 0;
    for (const auto& c : cases) {
        if (!check(c[0], c[1], rng)) ++failures;
    }
    return failures ? 1 : 0;
}
