// Test of the days_to_year CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu
//
// The reference is independent of the kernel's arithmetic: it tabulates the
// first day of every year 1..10000 by accumulating year lengths and looks
// each date up with a binary search.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* days, int* year, long long n);

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

static bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

// year_start[y] = days since 1970-01-01 of y-01-01, for y in 1..10000.
static std::vector<int> year_starts()
{
    std::vector<int> start(10001);
    start[1] = -719162;  // 0001-01-01
    for (int y = 1; y < 10000; ++y) start[y + 1] = start[y] + (is_leap(y) ? 366 : 365);
    return start;
}

static const int FIRST_DAY = -719162;  // 0001-01-01
static const int LAST_DAY = 2932896;   // 9999-12-31

static bool run_case(long long n, const std::vector<int>& start)
{
    std::vector<int> days(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i) days[i] = FIRST_DAY + (int)(next_u32() % (uint32_t)(LAST_DAY - FIRST_DAY + 1));
    // Boundary dates: first and last day of some years, and Feb 28 / Mar 1.
    const int years[] = {1, 2, 4, 100, 101, 400, 401, 1600, 1900, 1969, 1970, 2000, 2024, 2100, 2400, 9999};
    long long k = 0;
    for (int y : years) {
        const int feb28 = start[y] + 31 + 27;
        const int cand[4] = {start[y], feb28, feb28 + (is_leap(y) ? 2 : 1), (y < 10000 ? start[y + 1] : LAST_DAY + 1) - 1};
        for (int d : cand) if (k < n) days[k++] = d;
    }

    std::vector<int> expected(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i) {
        // Last year whose start is <= days[i].
        const int y = (int)(std::upper_bound(start.begin() + 1, start.end(), days[i]) - start.begin()) - 1;
        expected[i] = y;
    }

    int* d_days = nullptr;
    int* d_year = nullptr;
    CUDA_CHECK(cudaMalloc(&d_days, std::max<long long>(n, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_year, std::max<long long>(n, 1) * sizeof(int)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_days, days.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_year, 0, std::max<long long>(n, 1) * sizeof(int)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_days, d_year, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> year(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(year.data(), d_year, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_days));
    CUDA_CHECK(cudaFree(d_year));

    for (long long i = 0; i < n; ++i) {
        if (year[i] != expected[i]) {
            printf("FAIL n=%lld: day %d -> year %d, expected %d\n", n, days[i], year[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const std::vector<int> start = year_starts();
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        ok = run_case(n, start) && ok;
        printf("days_to_year cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
