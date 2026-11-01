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

static bool run_case(long long n, long long offset)
{
    const long long src_bytes = (offset + n + 7) / 8;
    const long long out_bytes = (n + 7) / 8;
    std::vector<unsigned char> source(src_bytes);
    for (auto& b : source) b = (unsigned char)next_u64();
    std::vector<unsigned char> out_init(out_bytes, 0xA5);

    unsigned char* d_source = to_device(source);
    unsigned char* d_out = to_device(out_init);
    kernel<<<blocks_for(out_bytes, 32), 256>>>(d_source, d_out, n, offset);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> out = to_host(d_out, out_bytes);
    CHECK(cudaFree(d_source));
    CHECK(cudaFree(d_out));

    std::vector<unsigned char> ref(out_bytes, 0);
    for (long long i = 0; i < n; ++i) {
        const long long j = offset + i;
        if ((source[j >> 3] >> (j & 7)) & 1) ref[i >> 3] |= (unsigned char)(1u << (i & 7));
    }
    for (long long k = 0; k < out_bytes; ++k) {
        if (out[k] != ref[k]) {
            printf("FAIL n=%lld offset=%lld byte %lld: got %02x want %02x\n", n, offset, k, out[k], ref[k]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long offsets[] = {0, 1, 3, 7, 8, 13, 64, 1000001};
    bool ok = true;
    for (long long n : sizes) {
        for (long long offset : offsets) ok = run_case(n, offset) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
