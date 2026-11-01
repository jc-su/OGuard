#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

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

static bool run_case(long long n, int mode)
{
    std::vector<long long> values(n);
    for (long long i = 0; i < n; ++i) {
        switch (mode) {
        case 0: values[i] = (long long)next_u64(); break;              // full range
        case 1: values[i] = (long long)(next_u64() % 50) - 25; break;  // many ties
        case 2: values[i] = 7; break;                                  // all equal
        case 3: values[i] = n - i; break;                              // decreasing
        default: values[i] = LLONG_MAX; break;                         // all maximal
        }
    }
    if (mode == 0 && n > 10) values[n - 3] = values[n / 2] = LLONG_MIN;
    std::vector<int> init(1, -1);

    long long* d_values = to_device(values);
    int* d_row = to_device(init);
    kernel<<<blocks_for(n, 8), 256>>>(d_values, d_row, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    const int got = to_host(d_row, 1)[0];
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_row));

    int want = -1;
    for (long long i = 0; i < n; ++i)
        if (want < 0 || values[i] < values[want]) want = (int)i;
    if (got != want) {
        printf("FAIL n=%lld mode=%d: got %d want %d\n", n, mode, got, want);
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        for (int mode = 0; mode < 5; ++mode) ok = run_case(n, mode) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
