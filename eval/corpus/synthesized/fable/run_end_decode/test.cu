// Test of the run_end_decode CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const double* run_values, const int* run_ends, double* out, long long r, long long n);

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

enum Shape { SINGLE, UNIT, SHORT, LONG };
static const char* shape_name(Shape s)
{
    switch (s) {
        case SINGLE: return "single";
        case UNIT: return "unit";
        case SHORT: return "short";
        default: return "long";
    }
}

// Run lengths that sum to n.
static std::vector<long long> make_runs(long long n, Shape shape)
{
    std::vector<long long> lengths;
    if (n == 0) return lengths;
    if (shape == SINGLE) { lengths.push_back(n); return lengths; }
    if (shape == UNIT) { lengths.assign(n, 1); return lengths; }
    const uint32_t high = shape == SHORT ? 7 : 2999;
    long long total = 0;
    while (total < n) {
        long long len = 1 + next_u32() % high;
        if (total + len > n) len = n - total;
        lengths.push_back(len);
        total += len;
    }
    return lengths;
}

static bool run_case(long long n, Shape shape)
{
    const std::vector<long long> lengths = make_runs(n, shape);
    const long long r = (long long)lengths.size();
    std::vector<int> run_ends(std::max<long long>(r, 1));
    std::vector<double> run_values(std::max<long long>(r, 1));
    std::vector<double> expected(std::max<long long>(n, 1));
    long long end = 0;
    for (long long j = 0; j < r; ++j) {
        run_values[j] = (double)(int32_t)next_u32() * 0.001;
        for (long long i = end; i < end + lengths[j]; ++i) expected[i] = run_values[j];
        end += lengths[j];
        run_ends[j] = (int)end;
    }

    double* d_values = nullptr;
    int* d_ends = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_values, std::max<long long>(r, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ends, std::max<long long>(r, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<long long>(n, 1) * sizeof(double)));
    if (r > 0) {
        CUDA_CHECK(cudaMemcpy(d_values, run_values.data(), r * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ends, run_ends.data(), r * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_out, 0xff, std::max<long long>(n, 1) * sizeof(double)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_values, d_ends, d_out, r, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> out(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_ends));
    CUDA_CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        if (out[i] != expected[i]) {
            printf("FAIL n=%lld shape=%s r=%lld: out[%lld] = %g, expected %g\n", n, shape_name(shape), r, i, out[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const Shape shapes[] = {SINGLE, UNIT, SHORT, LONG};
    bool ok = true;
    for (long long n : sizes) {
        for (Shape s : shapes) ok = run_case(n, s) && ok;
        printf("run_end_decode cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
