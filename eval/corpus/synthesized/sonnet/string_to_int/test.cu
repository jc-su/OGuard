#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <string>
#include <vector>

#include "kernel.cu"

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err_));                                 \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

// Reference parser with 128-bit arithmetic: returns false when the string does not parse.
static bool parse(const std::string &s, int64_t *value) {
    size_t p = 0;
    bool negative = false;
    if (p < s.size() && (s[p] == '+' || s[p] == '-')) {
        negative = s[p] == '-';
        p++;
    }
    if (p == s.size()) return false;
    __int128 acc = 0;
    for (; p < s.size(); p++) {
        if (s[p] < '0' || s[p] > '9') return false;
        acc = acc * 10 + (s[p] - '0');
        if (acc > ((__int128)1 << 64)) acc = (__int128)1 << 64;
    }
    if (negative) acc = -acc;
    if (acc < -((__int128)1 << 63) || acc > (((__int128)1 << 63) - 1)) return false;
    *value = (int64_t)acc;
    return true;
}

static int run_case(const char *name, const std::vector<std::string> &strings) {
    int64_t n = (int64_t)strings.size();
    std::vector<int32_t> offsets(n + 1, 0);
    std::vector<uint8_t> chars;
    for (int64_t i = 0; i < n; i++) {
        chars.insert(chars.end(), strings[i].begin(), strings[i].end());
        offsets[i + 1] = (int32_t)chars.size();
    }
    std::vector<int64_t> expected(n, 0);
    std::vector<uint8_t> expected_valid((n + 7) / 8, 0);
    for (int64_t i = 0; i < n; i++) {
        int64_t v;
        if (parse(strings[i], &v)) {
            expected[i] = v;
            expected_valid[i / 8] |= (uint8_t)(1u << (i % 8));
        }
    }
    int32_t *d_offsets;
    uint8_t *d_chars, *d_valid;
    int64_t *d_out;
    CHECK(cudaMalloc(&d_offsets, (n + 1) * 4));
    CHECK(cudaMalloc(&d_chars, chars.size() + 1));
    CHECK(cudaMalloc(&d_out, n * 8 + 1));
    CHECK(cudaMalloc(&d_valid, expected_valid.size() + 1));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chars, chars.data(), chars.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0x7B, n * 8 + 1));
    CHECK(cudaMemset(d_valid, 0, expected_valid.size() + 1));
    int64_t grid = (n + 255) / 256 > 0 ? (n + 255) / 256 : 1;
    kernel<<<(unsigned)grid, 256>>>(d_offsets, d_chars, d_out, d_valid, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> got(n);
    std::vector<uint8_t> got_valid(expected_valid.size());
    CHECK(cudaMemcpy(got.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_valid.data(), d_valid, expected_valid.size(), cudaMemcpyDeviceToHost));
    bool ok = got == expected && got_valid == expected_valid;
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    cudaFree(d_offsets);
    cudaFree(d_chars);
    cudaFree(d_out);
    cudaFree(d_valid);
    return ok ? 0 : 1;
}

static std::vector<std::string> random_strings(std::mt19937_64 &rng, int64_t n) {
    const std::string junk("0123456789+- _x.a\0\xff/:", 21);
    std::vector<std::string> result(n);
    for (auto &s : result) {
        int kind = (int)(rng() % 10);
        if (kind < 6) {
            int digits = 1 + (int)(rng() % 20);
            for (int k = 0; k < digits; k++) s += (char)('0' + rng() % 10);
            if (kind == 1) s = "-" + s;
            if (kind == 2) s = "+" + s;
            if (kind == 3) s = std::string(1 + rng() % 29, '0') + s;
        } else if (kind == 6) {
            s = "";
        } else if (kind == 7) {
            int len = 1 + (int)(rng() % 7);
            for (int k = 0; k < len; k++) s += junk[rng() % junk.size()];
        } else {
            s = std::to_string((int64_t)rng());
            if (kind == 9 && s.size() > 2) s[rng() % s.size()] = junk[rng() % junk.size()];
        }
    }
    return result;
}

int main() {
    std::mt19937_64 rng(5);
    int failed = 0;
    const int64_t sizes[] = {0, 1, 1000, 1000003};
    for (int64_t n : sizes) {
        char name[64];
        snprintf(name, sizeof name, "n=%lld", (long long)n);
        failed += run_case(name, random_strings(rng, n));
    }
    std::vector<std::string> specials = {
        "-9223372036854775808", "9223372036854775807", "-9223372036854775809", "9223372036854775808",
        "+9223372036854775807", "+9223372036854775808", "-9223372036854775807", "18446744073709551616",
        "99999999999999999999", "0", "-0", "+0", "1", "-1", "+1", "", "+", "-", " 1", "1 ", "1_0", "+-1",
        "-+1", "--1", "0x1", "1e5", "1.0", std::string("\xd9\xa1"), std::string("\0" "1", 2), "/1", ":1",
        std::string(100000, '0'), "-" + std::string(100000, '0') + "9223372036854775808",
        "+" + std::string(50000, '0') + "9223372036854775807", std::string(100000, '0') + "1" + std::string(5, '0'),
        std::string(25, '1'), "007", "-000000000000000000000000000012345"};
    failed += run_case("specials", specials);
    std::vector<std::string> repeated;
    for (size_t i = 0; i < 331; i++) repeated.push_back(specials[i % specials.size()]);
    failed += run_case("specials repeated (n not multiple of 8)", repeated);
    failed += run_case("all empty", std::vector<std::string>(1000, ""));
    std::vector<std::string> all_valid(1003);
    for (auto &s : all_valid) s = std::to_string((int64_t)rng());
    failed += run_case("all valid", all_valid);
    return failed ? 1 : 0;
}
