#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "kernel.cu"

#define CHECK(x)                                                                     \
    do {                                                                             \
        cudaError_t err_ = (x);                                                      \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64()
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

template <class T> static T* to_device(const std::vector<T>& h)
{
    T* d = nullptr;
    CHECK(cudaMalloc(&d, (h.size() > 0 ? h.size() : 1) * sizeof(T)));
    if (!h.empty()) CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <class T> static std::vector<T> to_host(const T* d, size_t n)
{
    std::vector<T> h(n);
    if (n) CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

[[maybe_unused]] static int sm_count()
{
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    return sms;
}

// Blocks of 256 threads for n items, at most blocks_per_sm per SM, at least 1.
[[maybe_unused]] static int blocks_for(long long n, int blocks_per_sm)
{
    long long blocks = (n + 255) / 256;
    const long long cap = (long long)sm_count() * blocks_per_sm;
    if (blocks > cap) blocks = cap;
    return blocks < 1 ? 1 : (int)blocks;
}

// Reference calendar built by counting days year by year.
static bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static int month_length(int y, int m)
{
    static const int len[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return (m == 2 && is_leap(y)) ? 29 : len[m - 1];
}

static std::vector<long long> month_start;  // days since 1970-01-01 of month index y * 12 + m - 1

static void build_calendar()
{
    month_start.assign(10000 * 12 + 1, 0);
    long long day = 0;  // days since 0001-01-01
    for (int y = 1; y <= 9999; ++y)
        for (int m = 1; m <= 12; ++m) {
            month_start[y * 12 + m - 1] = day;
            day += month_length(y, m);
        }
    month_start[10000 * 12] = day;
    const long long epoch = month_start[1970 * 12];
    for (int k = 12; k <= 10000 * 12; ++k) month_start[k] -= epoch;
}

static int reference(int days, int months)
{
    const long long k = std::upper_bound(month_start.begin() + 12, month_start.end(), (long long)days) -
                        month_start.begin() - 1;
    const int d = (int)(days - month_start[k]) + 1;
    const long long t = k + months;
    const int dim = month_length((int)(t / 12), (int)(t % 12) + 1);
    return (int)(month_start[t] + (d < dim ? d : dim) - 1);
}

static bool run_case(long long n, int max_months)
{
    const long long lo_day = month_start[12], hi_day = month_start[10000 * 12] - 1;
    std::vector<int> days(n), months(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        long long dd;
        if (r % 3 == 0) {  // last days of months, including February 29
            const long long k = 12 + (long long)((r >> 2) % (uint64_t)(9999 * 12));
            dd = month_start[k + 1] - 1 - (long long)((r >> 40) % 3);
        } else {
            dd = lo_day + (long long)((r >> 2) % (uint64_t)(hi_day - lo_day + 1));
        }
        days[i] = (int)dd;
        const long long k = std::upper_bound(month_start.begin() + 12, month_start.end(), dd) - month_start.begin() - 1;
        long long mlo = 12 - k, mhi = 9999 * 12 + 11 - k;
        if (mlo < -max_months) mlo = -max_months;
        if (mhi > max_months) mhi = max_months;
        months[i] = (int)(mlo + (long long)(next_u64() % (uint64_t)(mhi - mlo + 1)));
    }
    std::vector<int> out_init(n, 0x7fffffff);

    int* d_days = to_device(days);
    int* d_months = to_device(months);
    int* d_out = to_device(out_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_days, d_months, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int> out = to_host(d_out, n);
    CHECK(cudaFree(d_days));
    CHECK(cudaFree(d_months));
    CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        const int want = reference(days[i], months[i]);
        if (out[i] != want) {
            printf("FAIL n=%lld row %lld days %d months %d: got %d want %d\n", n, i, days[i], months[i], out[i], want);
            return false;
        }
    }
    return true;
}

int main()
{
    build_calendar();
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        ok = run_case(n, 120000) && ok;
        ok = run_case(n, 25) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
