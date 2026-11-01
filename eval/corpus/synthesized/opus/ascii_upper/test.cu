// Test the CUDA ascii_upper kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Each size runs with offsets[0] = 0 and with offsets[0] = 5, the second as
// for a sliced Arrow column. out starts as the complement of the expected
// bytes, so the test also fails when the kernel skips a byte of the column or
// writes a byte before offsets[0].
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

static void check(int64_t n, int32_t base, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int32_t> length(0, kMaxLength);
    std::vector<int32_t> offsets(n + 1);
    offsets[0] = base;
    for (int64_t i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + length(rng);
    const int64_t begin = offsets[0];
    const int64_t end = offsets[n];
    std::vector<uint8_t> chars(end);
    for (uint8_t& c : chars) c = static_cast<uint8_t>(rng());

    std::vector<uint8_t> before(end), expected(end);
    for (int64_t p = 0; p < end; ++p) {
        const uint8_t c = chars[p];
        const uint8_t upper = (c >= 'a' && c <= 'z') ? static_cast<uint8_t>(c - 32) : c;
        before[p] = static_cast<uint8_t>(~upper);
        expected[p] = p >= begin ? upper : before[p];
    }

    int32_t* d_offsets = to_device(offsets);
    uint8_t* d_chars = to_device(chars);
    uint8_t* d_out = to_device(before);

    // One thread per aligned 8-byte word that overlaps [begin, end).
    const int64_t words = (end + 7) / 8 - begin / 8;
    const int64_t blocks = (words + kThreads - 1) / kThreads;
    if (blocks > 0) {
        kernel<<<static_cast<unsigned>(blocks), kThreads>>>(d_offsets, d_chars, d_out, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint8_t> got = to_host(d_out, end);

    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_chars));
    CUDA_CHECK(cudaFree(d_out));

    if (!same_bytes(got, expected)) {
        std::fprintf(stderr, "ascii_upper: mismatch at n=%lld offsets[0]=%d\n",
                     static_cast<long long>(n), base);
        std::exit(1);
    }
    std::printf("ascii_upper n=%lld offsets[0]=%d passed\n", static_cast<long long>(n), base);
}

int main()
{
    std::mt19937_64 rng(4);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        for (const int32_t base : {0, 5}) {
            check(n, base, rng);
        }
    }
    return 0;
}
