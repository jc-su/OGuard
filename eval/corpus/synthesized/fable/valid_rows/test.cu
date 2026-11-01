// Test of the valid_rows CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const unsigned char* valid, int* rows, int* count, long long n);

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

static bool run_case(long long n, int density_percent)
{
    const long long nbytes = (n + 7) / 8;
    std::vector<unsigned char> valid(nbytes, 0);
    std::vector<int> expected;
    for (long long i = 0; i < n; ++i) {
        if ((int)(next_u32() % 100) < density_percent) {
            valid[i >> 3] |= (unsigned char)(1u << (i & 7));
            expected.push_back((int)i);
        }
    }

    unsigned char* d_valid = nullptr;
    int* d_rows = nullptr;
    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_valid, std::max<long long>(nbytes, 1) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_rows, std::max<long long>(n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    if (nbytes > 0) CUDA_CHECK(cudaMemcpy(d_valid, valid.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rows, 0xff, std::max<long long>(n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_valid, d_rows, d_count, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int count = -1;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> rows(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(rows.data(), d_rows, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_count));

    if (count != (int)expected.size()) {
        printf("FAIL n=%lld density=%d%%: count %d != %zu\n", n, density_percent, count, expected.size());
        return false;
    }
    rows.resize(count);
    std::sort(rows.begin(), rows.end());
    if (rows != expected) {
        printf("FAIL n=%lld density=%d%%: row ids differ\n", n, density_percent);
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const int densities[] = {0, 50, 97, 100};
    bool ok = true;
    for (long long n : sizes) {
        for (int d : densities) ok = run_case(n, d) && ok;
        printf("valid_rows cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
