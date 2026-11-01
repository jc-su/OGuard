// Test of the dictionary_decode CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* codes, const double* dictionary, double* out, long long n, long long d);

#define CUDA_CHECK(x)                                                                        \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t next_u32()
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static bool run_case(long long n, long long d)
{
    std::vector<int> codes(std::max<long long>(n, 1));
    std::vector<double> dictionary(d);
    for (long long j = 0; j < d; ++j) dictionary[j] = (double)(int32_t)next_u32() * 0.001;
    for (long long i = 0; i < n; ++i) codes[i] = (int)(next_u32() % (uint32_t)d);

    int* d_codes = nullptr;
    double* d_dict = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_codes, std::max<long long>(n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dict, d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<long long>(n, 1) * sizeof(double)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_codes, codes.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dict, dictionary.data(), d * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xff, std::max<long long>(n, 1) * sizeof(double)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_codes, d_dict, d_out, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_dict));
    CUDA_CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        if (out[i] != dictionary[codes[i]]) {
            printf("FAIL n=%lld d=%lld: out[%lld] = %g, expected %g\n", n, d, i, out[i], dictionary[codes[i]]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long dict_sizes[] = {1, 7, 4096};
    bool ok = true;
    for (long long n : sizes) {
        for (long long d : dict_sizes) ok = run_case(n, d) && ok;
        printf("dictionary_decode cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
