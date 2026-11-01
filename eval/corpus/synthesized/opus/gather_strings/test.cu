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

// Random string column: mostly short strings, some empty, a few long.
static void make_strings(long long n, std::vector<int>& offsets, std::vector<unsigned char>& chars)
{
    offsets.assign(n + 1, 0);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        long long len = (r % 10 == 0) ? 0 : (long long)(r >> 8) % 24;
        if (r % 4999 == 1) len = 3000 + (long long)(r >> 40) % 3000;
        offsets[i + 1] = offsets[i] + (int)len;
    }
    chars.resize(offsets[n]);
    for (auto& c : chars) c = (unsigned char)next_u64();
}

static bool run_case(long long n, long long m)
{
    std::vector<int> offsets;
    std::vector<unsigned char> chars;
    make_strings(n, offsets, chars);
    std::vector<int> idx(m), out_offsets(m + 1, 0);
    for (long long i = 0; i < m; ++i) {
        idx[i] = (int)(next_u64() % (uint64_t)n);
        out_offsets[i + 1] = out_offsets[i] + offsets[idx[i] + 1] - offsets[idx[i]];
    }
    const long long total = out_offsets[m];
    std::vector<unsigned char> out_init(total, 0xA5);

    int* d_offsets = to_device(offsets);
    unsigned char* d_chars = to_device(chars);
    int* d_idx = to_device(idx);
    int* d_out_offsets = to_device(out_offsets);
    unsigned char* d_out = to_device(out_init);
    kernel<<<sm_count() * 16, 256>>>(d_offsets, d_chars, d_idx, d_out_offsets, d_out, n, m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> out = to_host(d_out, total);
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_idx));
    CHECK(cudaFree(d_out_offsets));
    CHECK(cudaFree(d_out));

    std::vector<unsigned char> ref(total);
    for (long long i = 0; i < m; ++i)
        memcpy(ref.data() + out_offsets[i], chars.data() + offsets[idx[i]],
               offsets[idx[i] + 1] - offsets[idx[i]]);
    if (total && memcmp(out.data(), ref.data(), total) != 0) {
        for (long long b = 0; b < total; ++b)
            if (out[b] != ref[b]) {
                printf("FAIL n=%lld m=%lld byte %lld\n", n, m, b);
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
    for (long long s : sizes) {
        ok = run_case(s, s) && ok;
        if (s > 0) ok = run_case(s / 3 + 1, s) && ok;
        printf("n=%lld %s\n", s, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
