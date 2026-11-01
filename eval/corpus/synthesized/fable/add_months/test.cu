// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "kernel.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

template <class T> static T* upload(const std::vector<T>& v) {
    T* d; CHECK(cudaMalloc(&d, (v.size() + 1) * sizeof(T)));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Reference: a plain day-by-day calendar walk is too slow, so use a direct
// month table built from the Gregorian rules.
static bool leap(long long y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static int mdays(long long y, int m) {
    static const int t[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return m == 2 && leap(y) ? 29 : t[m - 1];
}
static long long days_from_ymd(long long y, int m, int d) {  // days since 1970-01-01
    long long cycles = (y - 1) / 400;
    long long days = cycles * 146097;
    for (long long yy = 1 + cycles * 400; yy < y; ++yy) days += leap(yy) ? 366 : 365;
    for (int mm = 1; mm < m; ++mm) days += mdays(y, mm);
    return days + d - 1 - 719162;
}
static void ymd_from_days(long long days, long long* y, int* m, int* d) {
    long long rem = days + 719162;
    long long yy = 1;
    long long per400 = 146097;
    yy += 400 * (rem / per400); rem %= per400;
    while (rem >= (leap(yy) ? 366 : 365)) { rem -= leap(yy) ? 366 : 365; ++yy; }
    int mm = 1;
    while (rem >= mdays(yy, mm)) { rem -= mdays(yy, mm); ++mm; }
    *y = yy; *m = mm; *d = (int)rem + 1;
}
static int reference(int days, int months) {
    long long y; int m, d;
    ymd_from_days(days, &y, &m, &d);
    long long total = y * 12 + (m - 1) + months;
    long long ty = total / 12; int tm = (int)(total % 12) + 1;
    int dd = std::min(d, mdays(ty, tm));
    return (int)days_from_ymd(ty, tm, dd);
}

static bool run_case(long long n, unsigned seed) {
    std::mt19937_64 rng(seed);
    const long long day_min = -719162, day_max = 2932896;  // 0001-01-01 and 9999-12-31
    std::vector<int> days(n), months(n), out(n, 7), expect(n);
    const int specials[] = {19782, 19388, -719162, 2932896, 11016, -25478};  // 2024-02-29, 2023-01-31, ... 1900-03-31
    for (long long i = 0; i < n; ++i) {
        if (i % 3 == 0) days[i] = (int)(9497 + rng() % 13514);  // 1996 to 2032
        else days[i] = (int)(day_min + (long long)(rng() % (unsigned long long)(day_max - day_min + 1)));
        if (i < 6) days[i] = specials[i];
        long long y; int m, d;
        ymd_from_days(days[i], &y, &m, &d);
        long long mo = (long long)(rng() % 240001) - 120000;
        long long idx = y * 12 + (m - 1) + mo;
        idx = std::max(12LL, std::min(idx, 9999LL * 12 + 11));
        months[i] = (int)(idx - (y * 12 + (m - 1)));
        expect[i] = reference(days[i], months[i]);
    }
    int* d_days = upload(days); int* d_months = upload(months); int* d_out = upload(out);
    long long blocks = (n + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_days, d_months, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != expect[i]) { ok = false; fprintf(stderr, "n=%lld row %lld: %d != %d (days %d months %d)\n", n, i, out[i], expect[i], days[i], months[i]); }
    cudaFree(d_days); cudaFree(d_months); cudaFree(d_out);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("add_months cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
