// Test for histogram: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const unsigned int* keys, int* counts, long long shift, long long n);

#define CHECK(call)                                                                    \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static const int THREADS = 256;
static const int MAX_BLOCKS = 256;  // small grid: the per-block flush is 256 global atomics
static int grid_for(long long n) {
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return (int)b;
}

static bool run_case(long long n, long long shift, bool timed) {
    std::vector<unsigned int> keys(n);
    std::vector<int> expected(256, 0);
    for (long long i = 0; i < n; ++i) {
        keys[i] = (unsigned int)next_u64();
        expected[(keys[i] >> shift) & 255u] += 1;
    }

    unsigned int* d_keys = nullptr;
    int* d_counts = nullptr;
    CHECK(cudaMalloc(&d_keys, std::max<long long>(n, 1) * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_counts, 256 * sizeof(int)));
    CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_counts, 0, 256 * sizeof(int)));

    kernel<<<grid_for(n), THREADS>>>(d_keys, d_counts, shift, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> counts(256);
    CHECK(cudaMemcpy(counts.data(), d_counts, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    const bool ok = std::equal(counts.begin(), counts.end(), expected.begin());
    printf("  n=%lld shift=%lld: %s\n", n, shift, ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_counts, 0, 256 * sizeof(int)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_keys, d_counts, shift, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_counts));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long shifts[] = {0, 8, 16, 24};
    bool ok = true;
    for (long long n : sizes)
        for (long long shift : shifts) ok = run_case(n, shift, n == 1000003 && shift == 8) && ok;
    printf("%s histogram\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
