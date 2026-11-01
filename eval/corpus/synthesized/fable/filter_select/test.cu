// Test for filter_select: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const double* quantity, double threshold, int* sel, int* count, long long n);

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

static bool run_case(long long n, bool timed) {
    std::vector<double> quantity(n);
    for (long long i = 0; i < n; ++i) quantity[i] = next_double() * 100.0;
    const double threshold = 50.0;
    std::vector<int> expected;
    for (long long i = 0; i < n; ++i)
        if (quantity[i] < threshold) expected.push_back((int)i);

    double* d_quantity = nullptr;
    int* d_sel = nullptr;
    int* d_count = nullptr;
    CHECK(cudaMalloc(&d_quantity, std::max<long long>(n, 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_sel, std::max<long long>(n, 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemcpy(d_quantity, quantity.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    kernel<<<grid_for(n), THREADS>>>(d_quantity, threshold, d_sel, d_count, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int count = 0;
    CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = (count == (int)expected.size());
    if (ok) {
        std::vector<int> sel(count);
        CHECK(cudaMemcpy(sel.data(), d_sel, count * sizeof(int), cudaMemcpyDeviceToHost));
        std::sort(sel.begin(), sel.end());
        ok = std::equal(sel.begin(), sel.end(), expected.begin());
    }
    printf("  n=%lld: %s (count=%d expected=%zu)\n", n, ok ? "ok" : "MISMATCH", count, expected.size());

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_count, 0, sizeof(int)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_quantity, threshold, d_sel, d_count, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_quantity));
    CHECK(cudaFree(d_sel));
    CHECK(cudaFree(d_count));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s filter_select\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
