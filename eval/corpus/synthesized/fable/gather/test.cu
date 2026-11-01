// Test for gather: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const double* col, const int* sel, double* out, long long m, long long n);

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
    const long long m = n;
    std::vector<double> col(n);
    std::vector<int> sel(m);
    for (long long i = 0; i < n; ++i) col[i] = next_double();
    for (long long i = 0; i < m; ++i) sel[i] = (int)(next_u64() % (uint64_t)n);
    std::vector<double> expected(m);
    for (long long i = 0; i < m; ++i) expected[i] = col[sel[i]];

    double* d_col = nullptr;
    int* d_sel = nullptr;
    double* d_out = nullptr;
    CHECK(cudaMalloc(&d_col, std::max<long long>(n, 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_sel, std::max<long long>(m, 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_out, std::max<long long>(m, 1) * sizeof(double)));
    CHECK(cudaMemcpy(d_col, col.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sel, sel.data(), m * sizeof(int), cudaMemcpyHostToDevice));

    kernel<<<grid_for(m), THREADS>>>(d_col, d_sel, d_out, m, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<double> out(m);
    CHECK(cudaMemcpy(out.data(), d_out, m * sizeof(double), cudaMemcpyDeviceToHost));
    bool ok = std::equal(out.begin(), out.end(), expected.begin());
    printf("  n=%lld: %s\n", n, ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(m), THREADS>>>(d_col, d_sel, d_out, m, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_col));
    CHECK(cudaFree(d_sel));
    CHECK(cudaFree(d_out));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s gather\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
