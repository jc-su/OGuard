#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

static bool run_case(long long n, long long unit)
{
    std::vector<long long> ts(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        switch (r % 4) {
        case 0: ts[i] = (long long)(next_u64() % (1ull << 63)) - (1ll << 62); break;  // [-2^62, 2^62)
        case 1: ts[i] = ((long long)(next_u64() % 2001) - 1000) * unit; break;         // multiples
        case 2: ts[i] = ((long long)(next_u64() % 2001) - 1000) * unit - 1; break;     // just below
        default: ts[i] = (long long)(next_u64() % 20000001) - 10000000; break;          // near the epoch
        }
    }
    if (n > 2) {
        ts[0] = -(1ll << 62);
        ts[1] = 1ll << 62;
    }
    std::vector<long long> out_init(n, 99);

    long long* d_ts = to_device(ts);
    long long* d_out = to_device(out_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_ts, d_out, n, unit);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    CHECK(cudaFree(d_ts));
    CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        const __int128 t = ts[i];
        __int128 q = t / unit;
        if (q * unit > t) q -= 1;
        const long long want = (long long)(q * unit);
        if (out[i] != want) {
            printf("FAIL n=%lld unit=%lld row %lld ts %lld: got %lld want %lld\n", n, unit, i, ts[i], out[i], want);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long units[] = {1, 7, 1000000, 3600000000ll, 86400000000ll, 1ll << 40};
    bool ok = true;
    for (long long n : sizes) {
        for (long long unit : units) ok = run_case(n, unit) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
