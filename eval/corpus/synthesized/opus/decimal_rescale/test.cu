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

#include <climits>

// Reference with 128-bit arithmetic.
static void reference(long long v, long long shift, long long& out, signed char& of)
{
    if (shift >= 0) {
        __int128 p = 1;
        for (int e = 0; e < shift; ++e) p *= 10;
        const __int128 prod = (__int128)v * p;
        if (prod > LLONG_MAX || prod < LLONG_MIN) { out = 0; of = 1; }
        else { out = (long long)prod; of = 0; }
    } else {
        __int128 d = 1;
        for (int e = 0; e < -shift; ++e) d *= 10;
        const __int128 a = v < 0 ? -(__int128)v : (__int128)v;
        __int128 q = a / d;
        if (2 * (a % d) >= d) q += 1;
        out = (long long)(v < 0 ? -q : q);
        of = 0;
    }
}

static bool run_case(long long n, long long shift)
{
    std::vector<long long> values(n);
    const long long specials[] = {LLONG_MIN, LLONG_MAX, 0, 1, -1, 5, -5, 15, -15, 25, -25,
                                  922337203685477580ll, -922337203685477580ll,
                                  922337203685477581ll, -922337203685477581ll, 500000000000000000ll,
                                  -500000000000000000ll, 449999999999999999ll};
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        switch (r % 4) {
        case 0: values[i] = (long long)next_u64(); break;
        case 1: values[i] = (long long)(next_u64() >> (r >> 2) % 64) * ((r & 64) ? -1 : 1); break;
        case 2: values[i] = specials[(r >> 8) % (sizeof(specials) / sizeof(specials[0]))]; break;
        default: values[i] = ((long long)(next_u64() % 2000001) - 1000000) * 5; break;
        }
    }
    std::vector<long long> out_init(n, 777);
    std::vector<signed char> of_init(n, 7);

    long long* d_values = to_device(values);
    long long* d_out = to_device(out_init);
    signed char* d_of = to_device(of_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_values, d_out, d_of, n, shift);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    std::vector<signed char> of = to_host(d_of, n);
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_of));

    for (long long i = 0; i < n; ++i) {
        long long want;
        signed char want_of;
        reference(values[i], shift, want, want_of);
        if (out[i] != want || of[i] != want_of) {
            printf("FAIL n=%lld shift=%lld row %lld value %lld: got %lld/%d want %lld/%d\n", n, shift, i,
                   values[i], out[i], of[i], want, want_of);
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
        for (long long shift = -18; shift <= 18; ++shift) ok = run_case(n, shift) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
