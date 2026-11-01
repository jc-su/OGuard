// Test the string_equal kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o string_equal_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* a_offsets,
                                  const unsigned char* a_chars,
                                  const int* b_offsets,
                                  const unsigned char* b_chars,
                                  signed char* match,
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

struct Column {
    std::vector<int> offsets;
    std::vector<unsigned char> chars;
};

// Two columns: 40% identical rows, 30% same length with one byte changed,
// 30% with a different length.
static void make_columns(long long n, std::mt19937_64& rng, Column& a, Column& b) {
    a.offsets.assign(n + 1, 0);
    b.offsets.assign(n + 1, 0);
    a.chars.clear();
    b.chars.clear();
    for (long long i = 0; i < n; ++i) {
        int a_len = (int)(rng() % 17);
        std::vector<unsigned char> s(a_len);
        for (int j = 0; j < a_len; ++j) s[j] = (unsigned char)(97 + rng() % 4);
        a.chars.insert(a.chars.end(), s.begin(), s.end());
        unsigned kind = (unsigned)(rng() % 10);
        if (kind >= 7) {
            int b_len = (int)(rng() % 17);
            for (int j = 0; j < b_len; ++j) b.chars.push_back((unsigned char)(97 + rng() % 4));
        } else {
            if (kind >= 4 && a_len > 0) s[rng() % a_len] ^= 1;
            b.chars.insert(b.chars.end(), s.begin(), s.end());
        }
        a.offsets[i + 1] = (int)a.chars.size();
        b.offsets[i + 1] = (int)b.chars.size();
    }
}

static bool check(long long n, std::mt19937_64& rng) {
    Column a, b;
    make_columns(n, rng, a, b);
    std::vector<signed char> expected(n);
    long long matches = 0;
    for (long long i = 0; i < n; ++i) {
        int la = a.offsets[i + 1] - a.offsets[i];
        int lb = b.offsets[i + 1] - b.offsets[i];
        bool eq = la == lb &&
                  std::memcmp(a.chars.data() + a.offsets[i], b.chars.data() + b.offsets[i], la) == 0;
        expected[i] = eq ? 1 : 0;
        matches += eq;
    }

    int *d_a_off, *d_b_off;
    unsigned char *d_a_chars, *d_b_chars;
    signed char* d_match;
    CUDA_CHECK(cudaMalloc(&d_a_off, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b_off, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a_chars, a.chars.size() + 1));
    CUDA_CHECK(cudaMalloc(&d_b_chars, b.chars.size() + 1));
    CUDA_CHECK(cudaMalloc(&d_match, n + 1));
    CUDA_CHECK(cudaMemcpy(d_a_off, a.offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_off, b.offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_chars, a.chars.data(), a.chars.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_chars, b.chars.data(), b.chars.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match, 0xFF, n + 1));

    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_a_off, d_a_chars, d_b_off, d_b_chars, d_match, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<signed char> match(n);
    CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
    bool ok = n == 0 || std::memcmp(match.data(), expected.data(), n) == 0;

    CUDA_CHECK(cudaFree(d_a_off));
    CUDA_CHECK(cudaFree(d_b_off));
    CUDA_CHECK(cudaFree(d_a_chars));
    CUDA_CHECK(cudaFree(d_b_chars));
    CUDA_CHECK(cudaFree(d_match));
    std::printf("n=%lld: %s (matches %lld of %lld)\n", n, ok ? "PASS" : "FAIL", matches, n);
    return ok;
}

int main() {
    std::mt19937_64 rng(3);
    const long long sizes[] = {0, 1, 1000, 1000003};
    int failures = 0;
    for (long long n : sizes) {
        if (!check(n, rng)) ++failures;
    }
    return failures ? 1 : 0;
}
