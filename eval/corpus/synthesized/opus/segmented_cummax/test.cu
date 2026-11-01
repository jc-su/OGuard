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

// Partition starts: mode 0 short partitions, 1 one partition, 2 every row, 3 long partitions.
static std::vector<int> make_first(long long n, int mode)
{
    std::vector<int> first(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        bool start = i == 0;
        if (mode == 0) start = start || r % 8 == 0;
        if (mode == 2) start = true;
        if (mode == 3) start = start || r % 300000 == 0;
        first[i] = start ? (int)i : first[i - 1];
    }
    return first;
}

static bool run_case(long long n, int mode)
{
    std::vector<int> first = make_first(n, mode);
    std::vector<long long> values(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        values[i] = (r % 64 == 0) ? LLONG_MIN : (r % 2 ? (long long)r : (long long)(r % 1000) - 500);
    }
    std::vector<long long> out_init(n, 99);

    long long* d_values = to_device(values);
    int* d_first = to_device(first);
    long long* d_out = to_device(out_init);
    void* args[] = {&d_values, &d_first, &d_out, &n};
    CHECK(cudaLaunchCooperativeKernel((const void*)kernel, cooperative_blocks((const void*)kernel, n), 256, args, 0, 0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_first));
    CHECK(cudaFree(d_out));

    long long m = 0;
    for (long long i = 0; i < n; ++i) {
        m = (first[i] == i || values[i] > m) ? values[i] : m;
        if (out[i] != m) {
            printf("FAIL n=%lld mode=%d row %lld: got %lld want %lld\n", n, mode, i, out[i], m);
            return false;
        }
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
