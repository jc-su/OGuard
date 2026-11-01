// Test the bloom_build kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o bloom_build_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const long long* keys,
                                  unsigned* bits,
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

static void reference(const std::vector<long long>& keys, long long words, std::vector<unsigned>& bits) {
    bits.assign(words, 0u);
    unsigned long long mask = (unsigned long long)(32 * words - 1);
    for (long long key : keys) {
        unsigned long long k = (unsigned long long)key;
        unsigned long long p1 = ((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        unsigned long long p2 = ((k * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
        bits[p1 >> 5] |= 1u << (p1 & 31);
        bits[p2 >> 5] |= 1u << (p2 & 31);
    }
}

static bool check(long long n, long long words, std::mt19937_64& rng) {
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) keys[i] = (long long)rng();
    std::vector<unsigned> expected;
    reference(keys, words, expected);

    long long* d_keys;
    unsigned* d_bits;
    CUDA_CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_bits, words * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bits, 0, words * sizeof(unsigned)));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_keys, d_bits, n, words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> bits(words);
    CUDA_CHECK(cudaMemcpy(bits.data(), d_bits, words * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool ok = std::memcmp(bits.data(), expected.data(), words * sizeof(unsigned)) == 0;
    long long set_bits = 0;
    for (unsigned w : expected) set_bits += __builtin_popcount(w);

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_bits));
    std::printf("n=%lld words=%lld: %s (%lld bits set)\n", n, words, ok ? "PASS" : "FAIL", set_bits);
    return ok;
}

int main() {
    std::mt19937_64 rng(6);
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
