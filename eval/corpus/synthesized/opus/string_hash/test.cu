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

static bool run_case(long long n, int max_len, bool offset_start)
{
    std::vector<int> offsets(n + 1);
    offsets[0] = offset_start ? 7 : 0;
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        int len = (r % 8 == 0) ? 0 : (int)((r >> 8) % (uint64_t)(max_len + 1));
        if (r % 4999 == 3) len = 5000;
        offsets[i + 1] = offsets[i] + len;
    }
    std::vector<unsigned char> chars(offsets[n]);
    for (auto& c : chars) c = (unsigned char)next_u64();
    std::vector<unsigned int> out_init(n, 0xDEADBEEFu);

    int* d_offsets = to_device(offsets);
    unsigned char* d_chars = to_device(chars);
    unsigned int* d_out = to_device(out_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_offsets, d_chars, d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned int> out = to_host(d_out, n);
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_out));

    for (long long i = 0; i < n; ++i) {
        uint32_t h = 2166136261u;
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) h = (h ^ chars[p]) * 16777619u;
        if (out[i] != h) {
            printf("FAIL n=%lld row %lld: got %08x want %08x\n", n, i, out[i], h);
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
        ok = run_case(n, 20, false) && ok;
        ok = run_case(n, 100, true) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
