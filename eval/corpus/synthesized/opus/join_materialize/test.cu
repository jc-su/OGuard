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

#include <algorithm>

static bool run_case(long long b, long long p, long long key_range, bool heavy)
{
    std::vector<long long> build(b), probe(p);
    for (auto& v : build) v = (long long)(next_u64() % (uint64_t)key_range);
    if (heavy) for (long long i = 0; i < b; i += 2) build[i] = 42;
    std::sort(build.begin(), build.end());
    for (auto& v : probe) v = (long long)(next_u64() % (uint64_t)(key_range + key_range / 4 + 1));
    if (heavy) for (long long i = 0; i < p; i += 10) probe[i] = 42;
    std::vector<int> offsets(p + 1, 0);
    for (long long i = 0; i < p; ++i) {
        auto range = std::equal_range(build.begin(), build.end(), probe[i]);
        offsets[i + 1] = offsets[i] + (int)(range.second - range.first);
    }
    const long long total = offsets[p];
    std::vector<int> init(total, -7);

    long long* d_build = to_device(build);
    long long* d_probe = to_device(probe);
    int* d_offsets = to_device(offsets);
    int* d_out_build = to_device(init);
    int* d_out_probe = to_device(init);
    kernel<<<sm_count() * 16, 256>>>(d_build, d_probe, d_offsets, d_out_build, d_out_probe, b, p);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int> out_build = to_host(d_out_build, total), out_probe = to_host(d_out_probe, total);
    CHECK(cudaFree(d_build));
    CHECK(cudaFree(d_probe));
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_out_build));
    CHECK(cudaFree(d_out_probe));

    for (long long i = 0; i < p; ++i) {
        const long long f = std::lower_bound(build.begin(), build.end(), probe[i]) - build.begin();
        for (long long j = 0; j < offsets[i + 1] - offsets[i]; ++j) {
            const long long q = offsets[i] + j;
            if (out_build[q] != f + j || out_probe[q] != i) {
                printf("FAIL b=%lld p=%lld position %lld: got (%d, %d) want (%lld, %lld)\n", b, p, q,
                       out_build[q], out_probe[q], f + j, i);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        ok = run_case(n, n, n + 1, false) && ok;
        ok = run_case(n, n, n / 8 + 1, false) && ok;
        ok = run_case(n, n / 5000 + 2, 1000, true) && ok;
        ok = run_case(n, n / 3 + 1, 4 * n + 1, false) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
