// Test of the string_contains CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* offsets, const unsigned char* chars, const unsigned char* pattern,
                                  signed char* match, long long n, long long k);

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

static const int ALPHABET = 3;  // bytes 'a', 'b', 'c'

static bool run_case(long long n, long long k)
{
    std::vector<int> offsets(n + 1);
    offsets[0] = 0;
    for (long long i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + (int)(next_u32() % 25);
    const long long total = offsets[n];
    std::vector<unsigned char> chars(std::max<long long>(total, 1));
    for (long long c = 0; c < total; ++c) chars[c] = (unsigned char)('a' + next_u32() % ALPHABET);
    std::vector<unsigned char> pattern(std::max<long long>(k, 1));
    for (long long j = 0; j < k; ++j) pattern[j] = (unsigned char)('a' + next_u32() % ALPHABET);

    std::vector<signed char> expected(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i) {
        const unsigned char* b = chars.data() + offsets[i];
        const unsigned char* e = chars.data() + offsets[i + 1];
        // The empty pattern occurs in every string, including the empty string.
        expected[i] = (k == 0 || std::search(b, e, pattern.data(), pattern.data() + k) != e) ? 1 : 0;
    }

    int* d_offsets = nullptr;
    unsigned char* d_chars = nullptr;
    unsigned char* d_pattern = nullptr;
    signed char* d_match = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chars, std::max<long long>(total, 1)));
    CUDA_CHECK(cudaMalloc(&d_pattern, std::max<long long>(k, 1)));
    CUDA_CHECK(cudaMalloc(&d_match, std::max<long long>(n, 1)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (total > 0) CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), total, cudaMemcpyHostToDevice));
    if (k > 0) CUDA_CHECK(cudaMemcpy(d_pattern, pattern.data(), k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match, 0xff, std::max<long long>(n, 1)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_offsets, d_chars, d_pattern, d_match, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<signed char> match(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_chars));
    CUDA_CHECK(cudaFree(d_pattern));
    CUDA_CHECK(cudaFree(d_match));

    for (long long i = 0; i < n; ++i) {
        if (match[i] != expected[i]) {
            printf("FAIL n=%lld k=%lld: match[%lld] = %d, expected %d\n", n, k, i, (int)match[i], (int)expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long pattern_lengths[] = {0, 1, 2, 3, 6};
    bool ok = true;
    for (long long n : sizes) {
        for (long long k : pattern_lengths) ok = run_case(n, k) && ok;
        printf("string_contains cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
