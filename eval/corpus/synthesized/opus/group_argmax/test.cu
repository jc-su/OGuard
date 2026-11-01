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

static bool run_case(long long n, long long g, bool sorted, bool ties, bool increasing)
{
    std::vector<int> group(n);
    std::vector<long long> value(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        group[i] = sorted ? (int)(i * g / (n > 0 ? n : 1)) : (int)(r % (uint64_t)g);
        if (increasing) value[i] = i;
        else value[i] = ties ? (long long)(next_u64() % 5) - 2 : (long long)next_u64();
    }
    std::vector<int> init(g, -1);

    int* d_group = to_device(group);
    long long* d_value = to_device(value);
    int* d_best = to_device(init);
    kernel<<<blocks_for(n, 32), 256>>>(d_group, d_value, d_best, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int> best = to_host(d_best, g);
    CHECK(cudaFree(d_group));
    CHECK(cudaFree(d_value));
    CHECK(cudaFree(d_best));

    std::vector<int> want(g, -1);
    for (long long i = 0; i < n; ++i) {
        int& w = want[group[i]];
        if (w < 0 || value[i] > value[w]) w = (int)i;
    }
    for (long long k = 0; k < g; ++k) {
        if (best[k] != want[k]) {
            printf("FAIL n=%lld g=%lld sorted=%d ties=%d group %lld: got %d want %d\n", n, g, (int)sorted, (int)ties,
                   k, best[k], want[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long groups[] = {1, 5, 1000, 1 << 20};
    bool ok = true;
    for (long long n : sizes) {
        for (long long g : groups) {
            ok = run_case(n, g, false, false, false) && ok;
            ok = run_case(n, g, false, true, false) && ok;
            ok = run_case(n, g, true, true, false) && ok;
            ok = run_case(n, g, true, false, true) && ok;
        }
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
