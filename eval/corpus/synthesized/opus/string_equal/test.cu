// Test the CUDA string_equal kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(err_));                      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kThreads = 256;
constexpr int kMaxLength = 16;

template <typename T>
static T* to_device(const std::vector<T>& host)
{
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, host.size() * sizeof(T)));
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return ptr;
}

template <typename T>
static std::vector<T> to_host(const T* ptr, size_t count)
{
    std::vector<T> host(count);
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host;
}

template <typename T>
static bool same_bytes(const std::vector<T>& got, const std::vector<T>& expected)
{
    return got.size() == expected.size() &&
           (got.empty() || std::memcmp(got.data(), expected.data(), got.size() * sizeof(T)) == 0);
}

struct Column {
    std::vector<int32_t> offsets;
    std::vector<uint8_t> chars;
};

// Column a is random. Row i of column b is, by kind: 0 a copy of a, 1 a with
// one byte changed, 2 a with its last byte removed or a byte appended, 3 a
// random string. A three-letter alphabet makes short random strings collide.
static void make_columns(int64_t n, std::mt19937_64& rng, Column& a, Column& b)
{
    std::uniform_int_distribution<int> length(0, kMaxLength);
    std::uniform_int_distribution<int> letter('a', 'c');
    std::uniform_int_distribution<int> kind_of(0, 3);
    a.offsets.assign(1, 0);
    b.offsets.assign(1, 0);
    a.chars.clear();
    b.chars.clear();
    for (int64_t i = 0; i < n; ++i) {
        const int a_len = length(rng);
        const size_t a_begin = a.chars.size();
        for (int k = 0; k < a_len; ++k) a.chars.push_back(static_cast<uint8_t>(letter(rng)));
        a.offsets.push_back(static_cast<int32_t>(a.chars.size()));

        const int kind = kind_of(rng);
        if (kind == 0) {
            for (int k = 0; k < a_len; ++k) b.chars.push_back(a.chars[a_begin + k]);
        } else if (kind == 1) {
            const size_t b_begin = b.chars.size();
            for (int k = 0; k < a_len; ++k) b.chars.push_back(a.chars[a_begin + k]);
            if (a_len > 0) {
                const int at = std::uniform_int_distribution<int>(0, a_len - 1)(rng);
                b.chars[b_begin + at] = static_cast<uint8_t>(b.chars[b_begin + at] + 1);
            }
        } else if (kind == 2) {
            const bool grow = a_len == 0 || (rng() & 1);
            const int b_len = grow ? a_len : a_len - 1;
            for (int k = 0; k < b_len; ++k) b.chars.push_back(a.chars[a_begin + k]);
            if (grow) b.chars.push_back(static_cast<uint8_t>(letter(rng)));
        } else {
            const int b_len = length(rng);
            for (int k = 0; k < b_len; ++k) b.chars.push_back(static_cast<uint8_t>(letter(rng)));
        }
        b.offsets.push_back(static_cast<int32_t>(b.chars.size()));
    }
}

static void check(int64_t n, std::mt19937_64& rng)
{
    Column a, b;
    make_columns(n, rng, a, b);

    std::vector<int8_t> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t a_len = a.offsets[i + 1] - a.offsets[i];
        const int32_t b_len = b.offsets[i + 1] - b.offsets[i];
        expected[i] = a_len == b_len &&
                      (a_len == 0 || std::memcmp(a.chars.data() + a.offsets[i],
                                                 b.chars.data() + b.offsets[i], a_len) == 0);
    }

    int32_t* d_a_offsets = to_device(a.offsets);
    uint8_t* d_a_chars = to_device(a.chars);
    int32_t* d_b_offsets = to_device(b.offsets);
    uint8_t* d_b_chars = to_device(b.chars);
    int8_t* d_match = nullptr;
    CUDA_CHECK(cudaMalloc(&d_match, n));
    CUDA_CHECK(cudaMemset(d_match, 0xFF, n));

    const int64_t blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_a_offsets, d_a_chars, d_b_offsets,
                                                             d_b_chars, d_match, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<int8_t> got = to_host(d_match, n);

    CUDA_CHECK(cudaFree(d_a_offsets));
    CUDA_CHECK(cudaFree(d_a_chars));
    CUDA_CHECK(cudaFree(d_b_offsets));
    CUDA_CHECK(cudaFree(d_b_chars));
    CUDA_CHECK(cudaFree(d_match));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "string_equal: mismatch at n=%lld\n", static_cast<long long>(n));
        std::exit(1);
    }
    std::printf("string_equal n=%lld passed\n", static_cast<long long>(n));
}

int main()
{
    std::mt19937_64 rng(3);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        check(n, rng);
    }
    return 0;
}
