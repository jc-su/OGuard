// Standalone test for starts_with. Builds a string column where about
// 30% of strings start with a fixed 3-byte prefix, launches the
// kernel, and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const int *offsets, const unsigned char *chars,
                                   const unsigned char *prefix, signed char *match,
                                   long long n, long long k);

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

const int BLOCK = 256;
const unsigned char PREFIX[] = {'a', 'b', 'c'};
const int K = 3;

static bool run_case(long long n, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> len_dist(0, 8);
    std::uniform_int_distribution<int> byte_dist('a', 'z');
    std::bernoulli_distribution force_prefix(0.3);

    std::vector<std::vector<unsigned char>> strings(n > 0 ? n : 0);
    std::vector<signed char> ref_match(n > 0 ? n : 1, 0);
    std::vector<int> offsets(n + 1, 0);
    std::vector<unsigned char> chars;

    for (long long i = 0; i < n; ++i) {
        int len = len_dist(rng);
        std::vector<unsigned char> s(len);
        for (int c = 0; c < len; ++c) s[c] = (unsigned char)byte_dist(rng);
        if (force_prefix(rng)) {
            for (int c = 0; c < K && c < len; ++c) s[c] = PREFIX[c];
        }
        strings[i] = s;

        bool matches = (len >= K) && (std::memcmp(s.data(), PREFIX, K) == 0);
        ref_match[i] = matches ? 1 : 0;

        chars.insert(chars.end(), s.begin(), s.end());
        offsets[i + 1] = (int)chars.size();
    }

    int *d_offsets;
    unsigned char *d_chars, *d_prefix;
    signed char *d_match;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&d_chars, sizeof(unsigned char) * (chars.size() > 0 ? chars.size() : 1)));
    CUDA_CHECK(cudaMalloc(&d_prefix, sizeof(unsigned char) * K));
    CUDA_CHECK(cudaMalloc(&d_match, sizeof(signed char) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * (n + 1),
                          cudaMemcpyHostToDevice));
    if (!chars.empty()) {
        CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), sizeof(unsigned char) * chars.size(),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_prefix, PREFIX, sizeof(unsigned char) * K,
                          cudaMemcpyHostToDevice));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_offsets, d_chars, d_prefix, d_match, n, K);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<signed char> got(n > 0 ? n : 1);
    CUDA_CHECK(cudaMemcpy(got.data(), d_match, sizeof(signed char) * n,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    long long matches = 0;
    for (long long i = 0; i < n; ++i) {
        if (got[i] != ref_match[i]) ok = false;
        matches += ref_match[i];
    }

    printf("n=%lld: %s (matches=%lld)\n", n, ok ? "PASS" : "FAIL", matches);

    cudaFree(d_offsets);
    cudaFree(d_chars);
    cudaFree(d_prefix);
    cudaFree(d_match);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);
    bool all_ok = true;
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        all_ok &= run_case(n, rng);
    }
    return all_ok ? 0 : 1;
}
