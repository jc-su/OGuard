// Test of the merge_sorted CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const long long* a, const long long* b, long long* out, long long m, long long p);

#define CUDA_CHECK(x)                                                                        \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64()
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static std::vector<long long> sorted_values(long long len, uint64_t range)
{
    std::vector<long long> v(std::max<long long>(len, 1));
    for (long long i = 0; i < len; ++i) v[i] = (long long)(next_u64() % (2 * range)) - (long long)range;
    std::sort(v.begin(), v.begin() + len);
    return v;
}

static bool run_case(long long m, long long p, uint64_t range)
{
    const std::vector<long long> a = sorted_values(m, range);
    const std::vector<long long> b = sorted_values(p, range);
    std::vector<long long> expected(std::max<long long>(m + p, 1));
    std::merge(a.begin(), a.begin() + m, b.begin(), b.begin() + p, expected.begin());

    long long* d_a = nullptr;
    long long* d_b = nullptr;
    long long* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, std::max<long long>(m, 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_b, std::max<long long>(p, 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<long long>(m + p, 1) * sizeof(long long)));
    if (m > 0) CUDA_CHECK(cudaMemcpy(d_a, a.data(), m * sizeof(long long), cudaMemcpyHostToDevice));
    if (p > 0) CUDA_CHECK(cudaMemcpy(d_b, b.data(), p * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0x80, std::max<long long>(m + p, 1) * sizeof(long long)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (m + p + 1023) / 1024);
    kernel<<<blocks, 256>>>(d_a, d_b, d_out, m, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> out(std::max<long long>(m + p, 1));
    if (m + p > 0) CUDA_CHECK(cudaMemcpy(out.data(), d_out, (m + p) * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    for (long long k = 0; k < m + p; ++k) {
        if (out[k] != expected[k]) {
            printf("FAIL m=%lld p=%lld range=%llu: out[%lld] = %lld, expected %lld\n", m, p, (unsigned long long)range, k, out[k], expected[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const uint64_t ranges[] = {50ull, 1ull << 62};  // many duplicates; nearly distinct
    bool ok = true;
    for (long long n : sizes) {
        const long long shapes[5][2] = {{n, n}, {n, n / 3}, {n / 3, n}, {n, 0}, {0, n}};
        for (const auto& s : shapes)
            for (uint64_t r : ranges) ok = run_case(s[0], s[1], r) && ok;
        printf("merge_sorted cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
