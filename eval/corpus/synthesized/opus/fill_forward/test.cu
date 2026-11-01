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

// Validity: mode 0 half valid, 1 all null, 2 one valid row per ~300000, 3 all valid.
static bool run_case(long long n, int mode)
{
    std::vector<long long> values(n);
    for (auto& v : values) v = (long long)next_u64();
    std::vector<unsigned char> valid((n + 7) / 8, 0);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        bool ok = mode == 3 || (mode == 0 && r % 2 == 0) || (mode == 2 && r % 300000 == 7);
        if (ok) valid[i >> 3] |= (unsigned char)(1u << (i & 7));
    }
    // Padding bits of the last byte are set; the kernel must not read them as rows.
    if (n % 8) valid[n / 8] |= (unsigned char)(0xFFu << (n % 8));
    std::vector<long long> out_init(n, 99);

    long long* d_values = to_device(values);
    unsigned char* d_valid = to_device(valid);
    long long* d_out = to_device(out_init);
    void* args[] = {&d_values, &d_valid, &d_out, &n};
    CHECK(cudaLaunchCooperativeKernel((const void*)kernel, cooperative_blocks((const void*)kernel, n), 256, args, 0, 0));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_valid));
    CHECK(cudaFree(d_out));

    long long cur = 0;
    for (long long i = 0; i < n; ++i) {
        if ((valid[i >> 3] >> (i & 7)) & 1) cur = values[i];
        if (out[i] != cur) {
            printf("FAIL n=%lld mode=%d row %lld: got %lld want %lld\n", n, mode, i, out[i], cur);
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
