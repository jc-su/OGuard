// Test of the radix_scatter CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const unsigned int* keys, int* cursors, int* out, long long n, long long shift);

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

enum Distribution { UNIFORM, SKEWED, SORTED };
static const char* distribution_name(Distribution d)
{
    return d == UNIFORM ? "uniform" : d == SKEWED ? "skewed" : "sorted";
}

static std::vector<unsigned int> make_keys(long long n, Distribution d)
{
    std::vector<unsigned int> keys(std::max<long long>(n, 1));
    static const unsigned int few[4] = {0u, 1u, 0x01020304u, 0xFFFFFFFFu};
    for (long long i = 0; i < n; ++i) keys[i] = d == SKEWED ? few[next_u32() & 3] : next_u32();
    if (d == SORTED) std::sort(keys.begin(), keys.begin() + n);
    return keys;
}

static bool run_case(long long n, long long shift, Distribution dist)
{
    const std::vector<unsigned int> keys = make_keys(n, dist);
    std::vector<long long> hist(256, 0);
    for (long long i = 0; i < n; ++i) ++hist[(keys[i] >> shift) & 255u];
    std::vector<int> cursors_in(256);
    long long acc = 0;
    for (int b = 0; b < 256; ++b) { cursors_in[b] = (int)acc; acc += hist[b]; }

    unsigned int* d_keys = nullptr;
    int* d_cursors = nullptr;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, std::max<long long>(n, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cursors, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<long long>(n, 1) * sizeof(int)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cursors, cursors_in.data(), 256 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xff, std::max<long long>(n, 1) * sizeof(int)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 2047) / 2048);
    kernel<<<blocks, 256>>>(d_keys, d_cursors, d_out, n, shift);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(std::max<long long>(n, 1));
    std::vector<int> cursors_out(256);
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cursors_out.data(), d_cursors, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_cursors));
    CUDA_CHECK(cudaFree(d_out));

    const char* tag = distribution_name(dist);
    // out is a permutation of 0..n-1 whose bytes are non-decreasing (grouped by bin in bin order).
    std::vector<char> seen(std::max<long long>(n, 1), 0);
    unsigned int prev = 0;
    for (long long k = 0; k < n; ++k) {
        const int id = out[k];
        if (id < 0 || id >= n || seen[id]) { printf("FAIL n=%lld shift=%lld %s: out[%lld] = %d is not a permutation entry\n", n, shift, tag, k, id); return false; }
        seen[id] = 1;
        const unsigned int b = (keys[id] >> shift) & 255u;
        if (b < prev) { printf("FAIL n=%lld shift=%lld %s: out is not grouped by byte at %lld\n", n, shift, tag, k); return false; }
        prev = b;
    }
    for (int b = 0; b < 256; ++b) {
        if (cursors_out[b] != cursors_in[b] + (int)hist[b]) {
            printf("FAIL n=%lld shift=%lld %s: cursor[%d] = %d, expected %lld\n", n, shift, tag, b, cursors_out[b], cursors_in[b] + hist[b]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long shifts[] = {0, 8, 16, 24};
    const Distribution dists[] = {UNIFORM, SKEWED, SORTED};
    bool ok = true;
    for (long long n : sizes) {
        for (Distribution d : dists)
            for (long long s : shifts) ok = run_case(n, s, d) && ok;
        printf("radix_scatter cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
