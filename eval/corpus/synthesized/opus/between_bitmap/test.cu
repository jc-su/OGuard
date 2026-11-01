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

static bool run_case(long long n, long long lo, long long hi)
{
    const long long nb = (n + 7) / 8;
    std::vector<long long> values(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        if (r % 97 == 0) values[i] = (r & 1) ? INT64_MAX : INT64_MIN;
        else values[i] = (long long)(r % 2001) - 1000;
    }
    std::vector<unsigned char> valid(nb);
    for (long long b = 0; b < nb; ++b) valid[b] = (unsigned char)next_u64();
    std::vector<unsigned char> out_init(nb, 0xA5);

    long long* d_values = to_device(values);
    unsigned char* d_valid = to_device(valid);
    unsigned char* d_out = to_device(out_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_values, d_valid, d_out, n, lo, hi);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> out = to_host(d_out, nb);
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_valid));
    CHECK(cudaFree(d_out));

    std::vector<unsigned char> ref(nb, 0);
    for (long long i = 0; i < n; ++i) {
        const bool ok = ((valid[i >> 3] >> (i & 7)) & 1) && lo <= values[i] && values[i] <= hi;
        if (ok) ref[i >> 3] |= (unsigned char)(1u << (i & 7));
    }
    if (memcmp(out.data(), ref.data(), nb) != 0) {
        for (long long b = 0; b < nb; ++b) {
            if (out[b] != ref[b]) {
                printf("FAIL n=%lld lo=%lld hi=%lld byte %lld: got %02x want %02x\n", n, lo, hi, b,
                       out[b], ref[b]);
                break;
            }
        }
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long bounds[][2] = {{-300, 450}, {INT64_MIN, INT64_MAX}, {5, -5}, {INT64_MIN, -1000}, {7, 7}};
    bool ok = true;
    for (long long n : sizes) {
        for (const auto& b : bounds) ok = run_case(n, b[0], b[1]) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
