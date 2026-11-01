// Test for segmented_sum: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// n is the number of values. Two average segment lengths per size exercise
// both kernel paths: about 8 values per segment (thread per segment) and
// about 128 (warp per segment). Segment boundaries are random cut points, so
// segments vary in length and some are empty. n=0 is also run with three
// empty segments.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int* offsets, const double* values, double* sums, long long g);

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
static double next_double() { return (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

static const int THREADS = 256;
static const int MAX_BLOCKS = 2048;
static int grid_for(long long n) {
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return (int)b;
}

static bool run_case(long long n, long long g, bool timed) {
    std::vector<int> offsets(g + 1);
    offsets[0] = 0;
    for (long long j = 1; j < g; ++j) offsets[j] = (int)(next_u64() % (uint64_t)(n + 1));
    std::sort(offsets.begin() + 1, offsets.begin() + std::max<long long>(g, 1));
    if (g > 0) offsets[g] = (int)n;
    std::vector<double> values(n);
    for (long long i = 0; i < n; ++i) values[i] = next_double();
    std::vector<double> expected(g);
    for (long long j = 0; j < g; ++j) {
        double acc = 0.0;
        for (int p = offsets[j]; p < offsets[j + 1]; ++p) acc += values[p];
        expected[j] = acc;
    }

    int* d_offsets = nullptr;
    double* d_values = nullptr;
    double* d_sums = nullptr;
    CHECK(cudaMalloc(&d_offsets, (g + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_values, std::max<long long>(n, 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_sums, std::max<long long>(g, 1) * sizeof(double)));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (g + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_sums, 0xff, std::max<long long>(g, 1) * sizeof(double)));

    kernel<<<grid_for(g), THREADS>>>(d_offsets, d_values, d_sums, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<double> sums(g);
    CHECK(cudaMemcpy(sums.data(), d_sums, g * sizeof(double), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long j = 0; j < g && ok; ++j)
        ok = std::fabs(sums[j] - expected[j]) <= 1e-9 * std::fabs(expected[j]) + 1e-9;
    printf("  n=%lld g=%lld: %s\n", n, g, ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(g), THREADS>>>(d_offsets, d_values, d_sums, g);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld g=%lld: %.1f us per launch\n", n, g, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_sums));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long avgs[] = {8, 128};
    bool ok = true;
    for (long long n : sizes) {
        if (n == 0) ok = run_case(0, 3, false) && ok;
        for (long long avg : avgs) {
            const long long g = n == 0 ? 0 : std::max<long long>(n / avg, 1);
            ok = run_case(n, g, n == 1000003) && ok;
        }
    }
    printf("%s segmented_sum\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
