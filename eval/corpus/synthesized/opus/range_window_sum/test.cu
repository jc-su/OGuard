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

// Cooperative grid for n items: all blocks resident, at most one block per 256 items.
static int cooperative_blocks(const void* fn, long long n)
{
    int per_sm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, 256, 0));
    long long blocks = (long long)per_sm * sm_count();
    const long long needed = (n + 255) / 256;
    if (blocks > needed) blocks = needed;
    return blocks < 1 ? 1 : (int)blocks;
}

// ts: mode 0 small gaps with duplicates, 1 all equal, 2 wide spread to +-2^62, 3 long runs of peers
static bool run_case(long long n, int mode, long long range)
{
    std::vector<long long> ts(n), values(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        switch (mode) {
        case 0: ts[i] = (long long)(r % (uint64_t)(4 * n + 1)) - 2 * n; break;
        case 1: ts[i] = -3; break;
        case 2: ts[i] = (long long)(r % (1ull << 63)) - (1ll << 62); break;
        default: ts[i] = (long long)(r % 7) * 1000; break;
        }
        values[i] = (long long)next_u64();
    }
    std::sort(ts.begin(), ts.end());
    if (mode == 2 && n > 2) {
        ts[0] = -(1ll << 62);
        ts[n - 1] = 1ll << 62;
    }
    std::vector<long long> out_init(n, 99);

    long long* d_ts = to_device(ts);
    long long* d_values = to_device(values);
    long long* d_out = to_device(out_init);
    void* args[] = {&d_ts, &d_values, &d_out, &n, &range};
    CHECK(cudaLaunchCooperativeKernel((const void*)kernel, cooperative_blocks((const void*)kernel, n), 256, args, 0, 0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    CHECK(cudaFree(d_ts));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_out));

    std::vector<unsigned long long> prefix(n + 1, 0);
    for (long long i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + (unsigned long long)values[i];
    for (long long i = 0; i < n; ++i) {
        const long long lo = std::lower_bound(ts.begin(), ts.end(), ts[i] - range) - ts.begin();
        const long long hi = std::upper_bound(ts.begin(), ts.end(), ts[i]) - ts.begin();
        const long long want = (long long)(prefix[hi] - prefix[lo]);
        if (out[i] != want) {
            printf("FAIL n=%lld mode=%d range=%lld row %lld: got %lld want %lld\n", n, mode, range, i, out[i], want);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long ranges[] = {0, 5, 1000, 1ll << 62};
    bool ok = true;
    for (long long n : sizes) {
        for (int mode = 0; mode < 4; ++mode)
            for (long long range : ranges) ok = run_case(n, mode, range) && ok;
        ok = run_case(n, 2, 1ll << 58) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
