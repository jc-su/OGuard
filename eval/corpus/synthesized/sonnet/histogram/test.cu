// Standalone test for histogram. Allocates inputs, launches the
// kernel, and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003} and shift in {0, 24}.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const unsigned int *keys, int *counts,
                                   long long shift, long long n);

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
const long long MAX_BLOCKS = 1024;

static bool run_case(long long n, long long shift, std::mt19937_64 &rng) {
    std::vector<unsigned int> keys(n > 0 ? n : 1);
    std::uniform_int_distribution<unsigned long long> dist(
        0, 0xFFFFFFFFULL);
    for (long long i = 0; i < n; ++i) keys[i] = (unsigned int)dist(rng);

    std::vector<int> ref(256, 0);
    for (long long i = 0; i < n; ++i) {
        int b = (int)((keys[i] >> shift) & 255U);
        ref[b]++;
    }

    unsigned int *d_keys;
    int *d_counts;
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(unsigned int) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * 256));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), sizeof(unsigned int) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * 256));

    if (n > 0) {
        long long blocks = std::min((n + BLOCK - 1) / BLOCK, MAX_BLOCKS);
        kernel<<<blocks, BLOCK>>>(d_keys, d_counts, shift, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> got(256);
    CUDA_CHECK(cudaMemcpy(got.data(), d_counts, sizeof(int) * 256,
                          cudaMemcpyDeviceToHost));

    bool ok = (got == ref);

    long long total = 0;
    for (int c : got) total += c;
    printf("n=%lld, shift=%lld: %s (total=%lld)\n", n, shift,
           ok ? "PASS" : "FAIL", total);

    cudaFree(d_keys);
    cudaFree(d_counts);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);
    bool all_ok = true;
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (long long shift : {0LL, 24LL}) {
            all_ok &= run_case(n, shift, rng);
        }
    }
    return all_ok ? 0 : 1;
}
