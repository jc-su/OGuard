// Test the CUDA tile_scan kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test test.cu
//
// Values span the whole int32 range, so the sums wrap around, and the
// reference wraps them the same way in unsigned 32-bit arithmetic. out and
// totals start as the complement of the expected values, so the test also
// fails when the kernel skips a row or a tile. kTile and kThreads come from
// kernel.cu.
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

template <typename T>
static std::vector<T> complement(const std::vector<T>& values)
{
    std::vector<T> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = static_cast<T>(~values[i]);
    return result;
}

static void check(int64_t n, std::mt19937_64& rng)
{
    const int64_t tiles = (n + kTile - 1) / kTile;
    std::vector<int32_t> values(n);
    for (int32_t& x : values) x = static_cast<int32_t>(static_cast<uint32_t>(rng()));

    std::vector<int32_t> expected_out(n), expected_totals(tiles);
    for (int64_t t = 0; t < tiles; ++t) {
        uint32_t sum = 0;
        for (int64_t i = t * kTile; i < n && i < (t + 1) * kTile; ++i) {
            expected_out[i] = static_cast<int32_t>(sum);
            sum += static_cast<uint32_t>(values[i]);
        }
        expected_totals[t] = static_cast<int32_t>(sum);
    }

    int32_t* d_values = to_device(values);
    int32_t* d_out = to_device(complement(expected_out));
    int32_t* d_totals = to_device(complement(expected_totals));

    if (tiles > 0) {
        kernel<<<static_cast<unsigned>(tiles), kThreads>>>(d_values, d_out, d_totals, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<int32_t> got_out = to_host(d_out, n);
    const std::vector<int32_t> got_totals = to_host(d_totals, tiles);

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_totals));

    if (!same_bytes(got_out, expected_out) || !same_bytes(got_totals, expected_totals)) {
        std::fprintf(stderr, "tile_scan: mismatch at n=%lld\n", static_cast<long long>(n));
        std::exit(1);
    }
    std::printf("tile_scan n=%lld passed\n", static_cast<long long>(n));
}

int main()
{
    std::mt19937_64 rng(8);
    for (const int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
        check(n, rng);
    }
    return 0;
}
