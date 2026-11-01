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

static bool run_case(long long n, int mode)
{
    std::vector<long long> values(n);
    long long v = 0;
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        switch (mode) {
        case 0: if (r % 4 == 0) v = (long long)(r >> 8) % 3; break;   // short runs, repeats possible
        case 1: v = 5; break;                                         // one run
        case 2: v = (long long)r; break;                              // all distinct (almost surely)
        default: if (r % 100000 == 0) v = (long long)(r >> 8); break; // long runs
        }
        values[i] = v;
    }
    std::vector<int> ends_init(n, -5);
    std::vector<long long> vals_init(n, -6);
    std::vector<int> count_init(1, 0);

    long long* d_values = to_device(values);
    int* d_run_ends = to_device(ends_init);
    long long* d_run_values = to_device(vals_init);
    int* d_count = to_device(count_init);
    void* args[] = {&d_values, &d_run_ends, &d_run_values, &d_count, &n};
    CHECK(cudaLaunchCooperativeKernel((const void*)kernel, cooperative_blocks((const void*)kernel, n), 256, args, 0, 0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int> run_ends = to_host(d_run_ends, n), count = to_host(d_count, 1);
    std::vector<long long> run_values = to_host(d_run_values, n);
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_run_ends));
    CHECK(cudaFree(d_run_values));
    CHECK(cudaFree(d_count));

    std::vector<int> want_ends(ends_init);
    std::vector<long long> want_vals(vals_init);
    int r = 0;
    for (long long i = 0; i < n; ++i) {
        if (i == n - 1 || values[i] != values[i + 1]) {
            want_ends[r] = (int)(i + 1);
            want_vals[r] = values[i];
            ++r;
        }
    }
    if (count[0] != r || run_ends != want_ends || run_values != want_vals) {
        printf("FAIL n=%lld mode=%d: count %d want %d\n", n, mode, count[0], r);
        for (long long j = 0; j < n; ++j)
            if (run_ends[j] != want_ends[j] || run_values[j] != want_vals[j]) {
                printf("  first difference at %lld: (%d, %lld) want (%d, %lld)\n", j, run_ends[j],
                       run_values[j], want_ends[j], want_vals[j]);
                break;
            }
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        for (int mode = 0; mode < 4; ++mode) ok = run_case(n, mode) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
