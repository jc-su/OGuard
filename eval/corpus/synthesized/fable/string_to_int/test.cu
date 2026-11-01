// nvcc -arch=sm_86 -O2 test.cu -o test && ./test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "kernel.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

template <class T> static T* upload(const std::vector<T>& v) {
    T* d; CHECK(cudaMalloc(&d, (v.size() + 1) * sizeof(T)));
    CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static const char* SPECIAL[] = {"", "+", "-", "0", "-0", "+0", "00000000000000000000000000012", "-9223372036854775808",
    "9223372036854775807", "9223372036854775808", "-9223372036854775809", "18446744073709551616",
    "99999999999999999999", " 12", "12 ", "1-2", "+-1", "--1", "1e3", "0x10", "12.0", "+9223372036854775807",
    "-00000009223372036854775808", "1111111111111111111111111111111111111111", "abc", "\xff" "5"};

// Reference: validate the syntax, then accumulate in __int128.
static bool reference(const std::string& s, long long* value) {
    size_t p = 0;
    bool neg = false;
    if (p < s.size() && (s[p] == '+' || s[p] == '-')) { neg = s[p] == '-'; ++p; }
    if (p >= s.size()) return false;
    __int128 mag = 0;
    for (; p < s.size(); ++p) {
        unsigned char c = s[p];
        if (c < '0' || c > '9') return false;
        mag = mag * 10 + (c - '0');
        if (mag > ((__int128)1 << 63)) return false;
    }
    __int128 v = neg ? -mag : mag;
    if (v > (__int128)0x7FFFFFFFFFFFFFFFll || v < -(__int128)0x7FFFFFFFFFFFFFFFll - 1) return false;
    *value = (long long)v;
    return true;
}

static bool run_case(long long n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> strings(n);
    for (long long i = 0; i < n; ++i) {
        switch (rng() % 6) {
            case 0: strings[i] = SPECIAL[rng() % (sizeof(SPECIAL) / sizeof(SPECIAL[0]))]; break;
            case 1: strings[i] = std::to_string((long long)rng()); break;
            case 2: strings[i] = std::string(rng() % 5, '0') + std::to_string(rng() % 1000000); break;
            case 3: strings[i] = "-" + std::string(rng() % 3, '0') + std::to_string(rng() % (1ull << 40)); break;
            case 4: { size_t len = rng() % 6; std::string s; for (size_t k = 0; k < len; ++k) s.push_back((char)(32 + rng() % 95)); strings[i] = s; break; }
            default: strings[i] = "+" + std::to_string(rng() % (1ull << 62)); break;
        }
    }
    std::vector<int> offsets(n + 1, 0);
    for (long long i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + (int)strings[i].size();
    std::vector<unsigned char> chars(offsets[n]);
    for (long long i = 0; i < n; ++i) memcpy(chars.data() + offsets[i], strings[i].data(), strings[i].size());
    long long nbytes = (n + 7) / 8;
    std::vector<long long> out(n, 7), eout(n, 0);
    std::vector<unsigned char> valid(nbytes, 0), evalid(nbytes, 0);
    for (long long i = 0; i < n; ++i) {
        long long v;
        if (reference(strings[i], &v)) { eout[i] = v; evalid[i / 8] |= (unsigned char)(1u << (i % 8)); }
    }
    int* d_offsets = upload(offsets); unsigned char* d_chars = upload(chars);
    long long* d_out = upload(out); unsigned char* d_valid = upload(valid);
    long long blocks = (nbytes + 255) / 256; if (blocks < 1) blocks = 1;
    kernel<<<(unsigned)blocks, 256>>>(d_offsets, d_chars, d_out, d_valid, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid.data(), d_valid, nbytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (long long i = 0; i < n && ok; ++i) if (out[i] != eout[i]) { ok = false; fprintf(stderr, "n=%lld row %lld '%s': %lld != %lld\n", n, i, strings[i].c_str(), out[i], eout[i]); }
    for (long long b = 0; b < nbytes && ok; ++b) if (valid[b] != evalid[b]) { ok = false; fprintf(stderr, "n=%lld valid byte %lld: %u != %u\n", n, b, valid[b], evalid[b]); }
    cudaFree(d_offsets); cudaFree(d_chars); cudaFree(d_out); cudaFree(d_valid);
    return ok;
}

int main() {
    long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, (unsigned)n + 1) && ok;
    printf("string_to_int cuda: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
