#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include "kernel.cu"

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(e_));                                 \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

// Reference by table lookup: days from 0001-01-01 to January 1st of each year,
// and cumulative days of each month. No shared formulas with the kernel.
static int64_t year_start[10002];
static const int mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static const int64_t EPOCH = 719162;  // days from 0001-01-01 to 1970-01-01

static bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static int dim_of(int y, int m) { return m == 2 && is_leap(y) ? 29 : mdays[m - 1]; }

static void init_tables() {
    year_start[1] = 0;
    for (int y = 1; y <= 10000; y++) year_start[y + 1] = year_start[y] + (is_leap(y) ? 366 : 365);
}

static void to_civil(int64_t days, int* y, int* m, int* d) {
    int64_t a = days + EPOCH;
    int lo = 1, hi = 9999;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (year_start[mid] <= a) lo = mid; else hi = mid - 1;
    }
    int64_t r = a - year_start[lo];
    int mm = 1;
    while (r >= dim_of(lo, mm)) { r -= dim_of(lo, mm); mm++; }
    *y = lo; *m = mm; *d = (int)r + 1;
}

static int64_t from_civil(int y, int m, int d) {
    int64_t a = year_start[y];
    for (int k = 1; k < m; k++) a += dim_of(y, k);
    return a + d - 1 - EPOCH;
}

static int32_t reference(int32_t days, int32_t months) {
    int y, m, d;
    to_civil(days, &y, &m, &d);
    int64_t t = (int64_t)y * 12 + (m - 1) + months;
    int ty = (int)(t / 12), tm = (int)(t % 12) + 1;
    return (int32_t)from_civil(ty, tm, std::min(d, dim_of(ty, tm)));
}

static uint64_t state = 88172645463325252ULL;
static uint64_t rnd() {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return state;
}

static bool run_case(int64_t n) {
    const int64_t MIN_DAY = -EPOCH, MAX_DAY = year_start[10000] - EPOCH - 1;
    std::vector<int32_t> days(n), months(n), expect(n), got(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t dd = MIN_DAY + (int64_t)(rnd() % (uint64_t)(MAX_DAY - MIN_DAY + 1));
        int y, m, d;
        to_civil(dd, &y, &m, &d);
        if (rnd() % 2) d = dim_of(y, m);
        if (i == 0) { y = 1; m = 1; d = 31; }
        if (i == 1) { y = 9999; m = 12; d = 31; }
        dd = from_civil(y, m, d);
        int64_t cur = (int64_t)y * 12 + (m - 1);
        int64_t lo = std::max<int64_t>(12, cur - 120000);
        int64_t hi = std::min<int64_t>(9999 * 12 + 11, cur + 120000);
        int64_t tgt = lo + (int64_t)(rnd() % (uint64_t)(hi - lo + 1));
        if (i == 0) tgt = cur;
        if (i == 1) tgt = cur;
        days[i] = (int32_t)dd;
        months[i] = (int32_t)(tgt - cur);
        expect[i] = reference(days[i], months[i]);
    }
    int32_t *d_days, *d_months, *d_out;
    size_t bytes = std::max<int64_t>(n, 1) * sizeof(int32_t);
    CHECK(cudaMalloc(&d_days, bytes));
    CHECK(cudaMalloc(&d_months, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_days, days.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_months, months.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, bytes));
    int64_t grid = std::max<int64_t>(1, (n + 255) / 256);
    kernel<<<(unsigned)grid, 256>>>(d_days, d_months, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_days));
    CHECK(cudaFree(d_months));
    CHECK(cudaFree(d_out));
    bool ok = n == 0 || memcmp(got.data(), expect.data(), n * sizeof(int32_t)) == 0;
    printf("add_months cuda n=%lld: %s\n", (long long)n, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    init_tables();
    bool ok = true;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) ok &= run_case(n);
    return ok ? 0 : 1;
}
