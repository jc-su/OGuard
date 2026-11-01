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

static bool run_case(long long n, long long r, long long tolerance, long long spread)
{
    std::vector<long long> left(n), right(r);
    for (auto& v : right) v = (long long)(next_u64() % (uint64_t)(2 * spread + 1)) - spread;
    std::sort(right.begin(), right.end());
    for (auto& v : left) v = (long long)(next_u64() % (uint64_t)(2 * spread + 1)) - spread;
    for (long long i = 0; i < n; i += 97) left[i] = (i & 1) ? (1ll << 61) : -(1ll << 61);
    std::vector<int> out_init(n, 12345);

    long long* d_left = to_device(left);
    long long* d_right = to_device(right);
    int* d_out = to_device(out_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_left, d_right, d_out, n, r, tolerance);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int> out = to_host(d_out, n);
    CHECK(cudaFree(d_left));
    CHECK(cudaFree(d_right));
    CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        const long long j = (long long)(std::upper_bound(right.begin(), right.end(), left[i]) - right.begin()) - 1;
        const int want = (j >= 0 && left[i] - right[j] <= tolerance) ? (int)j : -1;
        if (out[i] != want) {
            printf("FAIL n=%lld r=%lld tol=%lld row %lld: got %d want %d\n", n, r, tolerance, i, out[i], want);
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
        for (long long r : {0ll, 1ll, n / 2 + 1, 2 * n + 5}) {
            ok = run_case(n, r, 0, 1000) && ok;
            ok = run_case(n, r, 3, 1000) && ok;
            ok = run_case(n, r, 1ll << 62, 1ll << 61) && ok;
            ok = run_case(n, r, 1000000, 1ll << 40) && ok;
        }
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
