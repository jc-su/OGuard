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

static bool run_case(long long n, long long g, bool sorted)
{
    std::vector<int> group(n);
    std::vector<long long> value(n);
    std::vector<unsigned char> valid((n + 7) / 8);
    for (long long i = 0; i < n; ++i) {
        group[i] = sorted ? (int)(i * g / (n > 0 ? n : 1)) : (int)(next_u64() % (uint64_t)g);
        value[i] = (long long)next_u64();
    }
    for (auto& b : valid) b = (unsigned char)next_u64();
    std::vector<long long> zeros(g, 0);

    int* d_group = to_device(group);
    long long* d_value = to_device(value);
    unsigned char* d_valid = to_device(valid);
    long long* d_counts = to_device(zeros);
    long long* d_sums = to_device(zeros);
    kernel<<<blocks_for(n, 8), 256>>>(d_group, d_value, d_valid, d_counts, d_sums, n, g);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> counts = to_host(d_counts, g), sums = to_host(d_sums, g);
    CHECK(cudaFree(d_group));
    CHECK(cudaFree(d_value));
    CHECK(cudaFree(d_valid));
    CHECK(cudaFree(d_counts));
    CHECK(cudaFree(d_sums));

    std::vector<long long> rc(g, 0);
    std::vector<unsigned long long> rs(g, 0);
    for (long long i = 0; i < n; ++i) {
        if ((valid[i >> 3] >> (i & 7)) & 1) {
            rc[group[i]] += 1;
            rs[group[i]] += (unsigned long long)value[i];
        }
    }
    for (long long k = 0; k < g; ++k) {
        if (counts[k] != rc[k] || (unsigned long long)sums[k] != rs[k]) {
            printf("FAIL n=%lld g=%lld sorted=%d group %lld: count %lld/%lld sum %lld/%lld\n", n, g,
                   (int)sorted, k, counts[k], rc[k], sums[k], (long long)rs[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long groups[] = {1, 7, 1024};
    bool ok = true;
    for (long long n : sizes) {
        for (long long g : groups) {
            ok = run_case(n, g, false) && ok;
            ok = run_case(n, g, true) && ok;
        }
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
