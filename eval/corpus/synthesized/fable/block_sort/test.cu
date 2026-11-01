// Test the block_sort kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o block_sort_test kernel.cu test.cu
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* keys,
                                  int* out,
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

static const long long TILE = 1024;

// Full-range keys with 30% drawn from a small range for duplicates, and some
// INT_MIN and INT_MAX values.
static std::vector<int> make_keys(long long n, std::mt19937_64& rng) {
    std::vector<int> keys(n);
    for (long long i = 0; i < n; ++i) {
        unsigned pick = (unsigned)(rng() % 100);
        if (pick < 1) {
            keys[i] = (rng() % 2) ? INT_MIN : INT_MAX;
        } else if (pick < 31) {
            keys[i] = (int)(rng() % 11) - 5;
        } else {
            keys[i] = (int)(unsigned)rng();
        }
    }
    return keys;
}

static bool check(long long n, std::mt19937_64& rng) {
    std::vector<int> keys = make_keys(n, rng);
    long long tiles = (n + TILE - 1) / TILE;
    std::vector<int> expected(keys);
    for (long long t = 0; t < tiles; ++t) {
        long long end = std::min((t + 1) * TILE, n);
        std::sort(expected.begin() + t * TILE, expected.begin() + end);
    }

    int *d_keys, *d_out;
    CUDA_CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(int)));

    unsigned blocks = (unsigned)tiles;
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_keys, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = n == 0 || std::memcmp(out.data(), expected.data(), n * sizeof(int)) == 0;

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("n=%lld: %s (%lld tiles)\n", n, ok ? "PASS" : "FAIL", tiles);
    return ok;
}

int main() {
    std::mt19937_64 rng(9);
    const long long sizes[] = {0, 1, 1000, 1024, 1025, 1000003};
    int failures = 0;
    for (long long n : sizes) {
        if (!check(n, rng)) ++failures;
    }
    return failures ? 1 : 0;
}
