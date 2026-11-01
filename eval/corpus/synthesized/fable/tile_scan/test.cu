// Test the tile_scan kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o tile_scan_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* values,
                                  int* out,
                                  int* totals,
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

static bool check(long long n, std::mt19937_64& rng) {
    std::vector<int> values(n);
    for (long long i = 0; i < n; ++i) values[i] = (int)(rng() % 2001) - 1000;
    long long tiles = (n + TILE - 1) / TILE;
    std::vector<int> expected_out(n), expected_totals(tiles);
    for (long long t = 0; t < tiles; ++t) {
        int running = 0;
        for (long long i = t * TILE; i < (t + 1) * TILE && i < n; ++i) {
            expected_out[i] = running;
            running += values[i];
        }
        expected_totals[t] = running;
    }

    int *d_values, *d_out, *d_totals;
    CUDA_CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_totals, (tiles + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_totals, 0xFF, (tiles + 1) * sizeof(int)));

    unsigned blocks = (unsigned)tiles;
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_values, d_out, d_totals, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(n), totals(tiles);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(totals.data(), d_totals, tiles * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = (n == 0 || std::memcmp(out.data(), expected_out.data(), n * sizeof(int)) == 0) &&
              (tiles == 0 || std::memcmp(totals.data(), expected_totals.data(), tiles * sizeof(int)) == 0);

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_totals));
    std::printf("n=%lld: %s (%lld tiles)\n", n, ok ? "PASS" : "FAIL", tiles);
    return ok;
}

int main() {
    std::mt19937_64 rng(8);
    const long long sizes[] = {0, 1, 1000, 1024, 1025, 1000003};
    int failures = 0;
    for (long long n : sizes) {
        if (!check(n, rng)) ++failures;
    }
    return failures ? 1 : 0;
}
