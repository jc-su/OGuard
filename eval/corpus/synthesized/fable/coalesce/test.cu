// Test the coalesce kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o coalesce_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* a,
                                  const unsigned char* valid,
                                  const double* b,
                                  double* out,
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

static bool check(long long n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    std::vector<double> a(n), b(n);
    long long bytes = (n + 7) / 8;
    std::vector<unsigned char> valid(bytes, 0);
    for (long long i = 0; i < n; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        if ((rng() % 10) < 6) valid[i >> 3] |= (unsigned char)(1u << (i & 7));
    }
    std::vector<double> expected(n);
    for (long long i = 0; i < n; ++i) {
        expected[i] = ((valid[i >> 3] >> (i & 7)) & 1) ? a[i] : b[i];
    }

    double *d_a, *d_b, *d_out;
    unsigned char* d_valid;
    CUDA_CHECK(cudaMalloc(&d_a, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_valid, bytes + 1));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valid, valid.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(double)));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_a, d_valid, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    bool ok = n == 0 || std::memcmp(out.data(), expected.data(), n * sizeof(double)) == 0;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_valid));
    std::printf("n=%lld: %s\n", n, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937_64 rng(1);
    const long long sizes[] = {0, 1, 1000, 1000003};
    int failures = 0;
    for (long long n : sizes) {
        if (!check(n, rng)) ++failures;
    }
    return failures ? 1 : 0;
}
