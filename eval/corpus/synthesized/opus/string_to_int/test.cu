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

#include <string>

static std::string make_string()
{
    static const char* fixed[] = {"0", "-0", "+0", "123", "-123", "+00042", "9223372036854775807",
                                  "-9223372036854775808", "9223372036854775808", "-9223372036854775809",
                                  "+9223372036854775807", "18446744073709551616", "99999999999999999999",
                                  "", "+", "-", "12a", " 12", "1 ", "--1", "+-1", "0x10", "1e5", "-00000000000000000000000000009223372036854775808"};
    const uint64_t r = next_u64();
    std::string s;
    switch (r % 6) {
    case 0: return fixed[(r >> 8) % (sizeof(fixed) / sizeof(fixed[0]))];
    case 1: case 2: {  // random number with optional sign and leading zeros
        if ((r >> 8) % 3 == 1) s += '-';
        if ((r >> 8) % 3 == 2) s += '+';
        s.append((r >> 16) % 4, '0');
        const int digits = 1 + (int)((r >> 24) % 20);
        for (int k = 0; k < digits; ++k) s += (char)('0' + next_u64() % 10);
        return s;
    }
    case 3: {  // long run of leading zeros
        s.assign(200 + (r >> 8) % 300, '0');
        s += std::to_string((long long)(next_u64() >> 1) * ((r & 64) ? -1 : 1)).substr((r & 64) ? 1 : 0);
        if (r & 64) s = "-" + s;
        return s;
    }
    case 4: {  // one corrupted byte
        s = std::to_string((long long)next_u64());
        s[(r >> 8) % s.size()] = (char)(next_u64() & 0xff);
        return s;
    }
    default: {  // random bytes
        const int len = (int)((r >> 8) % 6);
        for (int k = 0; k < len; ++k) s += (char)next_u64();
        return s;
    }
    }
}

static bool reference(const std::string& s, long long& value)
{
    size_t p = 0;
    bool neg = false;
    if (p < s.size() && (s[p] == '+' || s[p] == '-')) neg = s[p++] == '-';
    if (p >= s.size()) return false;
    __int128 mag = 0;
    for (; p < s.size(); ++p) {
        if (s[p] < '0' || s[p] > '9') return false;
        mag = mag * 10 + (s[p] - '0');
        if (mag > ((__int128)1 << 64)) mag = ((__int128)1 << 64);  // saturate, already too large
    }
    const __int128 v = neg ? -mag : mag;
    if (v < -((__int128)1 << 63) || v > ((__int128)1 << 63) - 1) return false;
    value = (long long)v;
    return true;
}

static bool run_case(long long n)
{
    std::vector<std::string> strs(n);
    std::vector<int> offsets(n + 1, 0);
    for (long long i = 0; i < n; ++i) {
        strs[i] = make_string();
        offsets[i + 1] = offsets[i] + (int)strs[i].size();
    }
    std::vector<unsigned char> chars(offsets[n]);
    for (long long i = 0; i < n; ++i) memcpy(chars.data() + offsets[i], strs[i].data(), strs[i].size());
    std::vector<long long> out_init(n, 77);
    std::vector<unsigned char> valid_init((n + 7) / 8, 0);

    int* d_offsets = to_device(offsets);
    unsigned char* d_chars = to_device(chars);
    long long* d_out = to_device(out_init);
    unsigned char* d_valid = to_device(valid_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_offsets, d_chars, d_out, d_valid, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> out = to_host(d_out, n);
    std::vector<unsigned char> valid = to_host(d_valid, (n + 7) / 8);
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_valid));

    std::vector<unsigned char> want_valid((n + 7) / 8, 0);
    for (long long i = 0; i < n; ++i) {
        long long v = 0;
        const bool ok = reference(strs[i], v);
        if (ok) want_valid[i >> 3] |= (unsigned char)(1u << (i & 7));
        if (out[i] != (ok ? v : 0) || (bool)((valid[i >> 3] >> (i & 7)) & 1) != ok) {
            printf("FAIL n=%lld row %lld \"%s\": got %lld/%d want %lld/%d\n", n, i, strs[i].c_str(), out[i],
                   (valid[i >> 3] >> (i & 7)) & 1, ok ? v : 0, (int)ok);
            return false;
        }
    }
    if (valid != want_valid) {
        printf("FAIL n=%lld: padding bits of the bitmap are set\n", n);
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        for (int rep = 0; rep < 2; ++rep) ok = run_case(n) && ok;
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
