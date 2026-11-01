// Test the bloom_probe kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o bloom_probe_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const long long* keys,
                                  const unsigned* bits,
                                  signed char* match,
                                  long long n,
                                  long long words);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err));                              \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static void hash_positions(long long key, long long words, unsigned long long& p1, unsigned long long& p2) {
    unsigned long long k = (unsigned long long)key;
    unsigned long long mask = (unsigned long long)(32 * words - 1);
    p1 = ((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
    p2 = ((k * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
}

static bool check(long long n, long long words, std::mt19937_64& rng) {
    // Build the filter on the CPU from the member keys, then probe a mix of
    // members and random keys.
    long long member_count = n / 2 > 0 ? n / 2 : 1;
    std::vector<long long> members(member_count);
    for (long long& m : members) m = (long long)rng();
    std::vector<unsigned> bits(words, 0u);
    for (long long m : members) {
        unsigned long long p1, p2;
        hash_positions(m, words, p1, p2);
        bits[p1 >> 5] |= 1u << (p1 & 31);
        bits[p2 >> 5] |= 1u << (p2 & 31);
    }
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) {
        keys[i] = (rng() % 2) ? members[rng() % member_count] : (long long)rng();
    }
    std::vector<signed char> expected(n);
    long long matches = 0;
    for (long long i = 0; i < n; ++i) {
        unsigned long long p1, p2;
        hash_positions(keys[i], words, p1, p2);
        bool hit = ((bits[p1 >> 5] >> (p1 & 31)) & 1u) && ((bits[p2 >> 5] >> (p2 & 31)) & 1u);
        expected[i] = hit ? 1 : 0;
        matches += hit;
    }

    long long* d_keys;
    unsigned* d_bits;
    signed char* d_match;
    CUDA_CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_bits, words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_match, n + 1));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bits, bits.data(), words * sizeof(unsigned), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match, 0xFF, n + 1));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_keys, d_bits, d_match, n, words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<signed char> match(n);
    CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
    bool ok = n == 0 || std::memcmp(match.data(), expected.data(), n) == 0;

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_match));
    std::printf("n=%lld words=%lld: %s (%lld matches of %lld)\n", n, words, ok ? "PASS" : "FAIL", matches, n);
    return ok;
}

int main() {
    std::mt19937_64 rng(7);
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long words_list[] = {1, 1 << 10, 1 << 20, 1 << 26};
    int failures = 0;
    for (long long n : sizes) {
        for (long long words : words_list) {
            if (!check(n, words, rng)) ++failures;
        }
    }
    return failures ? 1 : 0;
}
