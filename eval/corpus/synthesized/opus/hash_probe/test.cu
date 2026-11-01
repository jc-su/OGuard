// GPU test of kernel.cu against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_hash_probe

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                                  \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(status));                                 \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

constexpr long long kGuard = 64;
constexpr int kUnset = -7;

static void check(bool ok, const char* what, long long m)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL m=%lld: %s\n", m, what);
        std::exit(1);
    }
}

static unsigned long long fmix64(unsigned long long h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static unsigned long long inverse_odd(unsigned long long c)
{
    unsigned long long x = c;  // Newton iteration doubles the correct low bits.
    for (int i = 0; i < 6; ++i) {
        x *= 2 - c * x;
    }
    return x;
}

static long long fmix64_inverse(unsigned long long x)
{
    x ^= x >> 33;
    x *= inverse_odd(0xc4ceb9fe1a85ec53ULL);
    x ^= x >> 33;
    x *= inverse_odd(0xff51afd7ed558ccdULL);
    x ^= x >> 33;
    return static_cast<long long>(x);
}

static long long table_capacity(long long n)
{
    long long capacity = 1;
    while (capacity < 2 * n) {
        capacity *= 2;
    }
    return capacity;
}

static int default_grid(long long n, int block)
{
    int device = 0;
    int sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const long long blocks = (n + block - 1) / block;
    return static_cast<int>(std::max(1LL, std::min(blocks, 8LL * sms)));
}

struct Table {
    std::vector<long long> keys;
    std::vector<int> rows;
};

// The table that hash_build writes, built on the CPU.
static Table build_table(const std::vector<long long>& build_keys, long long capacity)
{
    Table table{std::vector<long long>(capacity, -1), std::vector<int>(capacity, 0)};
    const unsigned long long mask = static_cast<unsigned long long>(capacity - 1);
    for (size_t i = 0; i < build_keys.size(); ++i) {
        unsigned long long slot = fmix64(static_cast<unsigned long long>(build_keys[i])) & mask;
        while (table.keys[slot] != -1) {
            slot = (slot + 1) & mask;
        }
        table.keys[slot] = build_keys[i];
        table.rows[slot] = static_cast<int>(i);
    }
    return table;
}

// Build row of each probe row, or -1 when the probe key has no match.
static std::vector<int> expected_build_rows(const std::vector<long long>& build_keys,
                                            const std::vector<long long>& probe_keys)
{
    std::unordered_map<long long, int> row_of;
    for (size_t i = 0; i < build_keys.size(); ++i) {
        row_of[build_keys[i]] = static_cast<int>(i);
    }
    std::vector<int> rows(probe_keys.size(), -1);
    for (size_t j = 0; j < probe_keys.size(); ++j) {
        const auto it = row_of.find(probe_keys[j]);
        if (it != row_of.end()) {
            rows[j] = it->second;
        }
    }
    return rows;
}

static void probe_and_check(const std::vector<long long>& build_keys,
                            const std::vector<long long>& probe_keys, long long out_cap, int grid,
                            int block)
{
    const long long m = static_cast<long long>(probe_keys.size());
    const long long capacity = table_capacity(static_cast<long long>(build_keys.size()));
    const Table table = build_table(build_keys, capacity);

    long long* d_probe_keys = nullptr;
    long long* d_table_keys = nullptr;
    int* d_table_rows = nullptr;
    int* d_out_build = nullptr;
    int* d_out_probe = nullptr;
    int* d_out_count = nullptr;
    const std::vector<int> unset(out_cap + kGuard, kUnset);
    CUDA_CHECK(cudaMalloc(&d_probe_keys, m * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_table_rows, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_build, unset.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_probe, unset.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probe_keys, probe_keys.data(), m * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_keys, table.keys.data(), capacity * sizeof(long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_rows, table.rows.data(), capacity * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_build, unset.data(), unset.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_probe, unset.data(), unset.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    kernel<<<grid, block>>>(d_probe_keys, d_table_keys, d_table_rows, capacity, d_out_build,
                            d_out_probe, d_out_count, m, out_cap);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out_build(unset.size());
    std::vector<int> out_probe(unset.size());
    int count = 0;
    CUDA_CHECK(cudaMemcpy(out_build.data(), d_out_build, unset.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_probe.data(), d_out_probe, unset.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_probe_keys));
    CUDA_CHECK(cudaFree(d_table_keys));
    CUDA_CHECK(cudaFree(d_table_rows));
    CUDA_CHECK(cudaFree(d_out_build));
    CUDA_CHECK(cudaFree(d_out_probe));
    CUDA_CHECK(cudaFree(d_out_count));

    const std::vector<int> want = expected_build_rows(build_keys, probe_keys);
    const long long matches = std::count_if(want.begin(), want.end(), [](int r) { return r >= 0; });
    check(count == matches, "out_count differs from the number of matches", m);
    const long long written = std::min(static_cast<long long>(count), out_cap);
    std::vector<bool> seen(m, false);
    for (long long p = 0; p < written; ++p) {
        const int j = out_probe[p];
        check(j >= 0 && j < m, "probe row out of range", m);
        check(!seen[j], "a probe row appears twice", m);
        seen[j] = true;
        check(want[j] >= 0 && out_build[p] == want[j], "wrong pair", m);
    }
    for (size_t p = written; p < unset.size(); ++p) {
        check(out_build[p] == kUnset && out_probe[p] == kUnset, "write past the pairs", m);
    }
}

// Distinct keys: an odd multiplier is a bijection on 64-bit integers.
static std::vector<long long> distinct_keys(long long n, unsigned long long offset,
                                            std::mt19937_64& rng)
{
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) {
        keys[i] = static_cast<long long>((static_cast<unsigned long long>(i) + offset) *
                                         0x9e3779b97f4a7c15ULL);
    }
    std::shuffle(keys.begin(), keys.end(), rng);
    return keys;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const int block = 256;
    std::mt19937_64 rng(6);
    for (const long long n : sizes) {
        const std::vector<long long> build_keys = distinct_keys(n, 12345, rng);
        // Probe row 0 and about half the others repeat build keys; one is -1.
        std::vector<long long> probe_keys = distinct_keys(n, 1ULL << 40, rng);
        for (long long j = 0; j < n; ++j) {
            if (j == 0 || rng() % 2 == 0) {
                probe_keys[j] = build_keys[rng() % n];
            }
        }
        if (n >= 3) {
            probe_keys[n / 3] = -1;
        }
        const std::vector<int> want = expected_build_rows(build_keys, probe_keys);
        const long long matches =
            std::count_if(want.begin(), want.end(), [](int r) { return r >= 0; });
        probe_and_check(build_keys, probe_keys, n, default_grid(n, block), block);
        probe_and_check(build_keys, probe_keys, matches / 2, default_grid(n, block), block);
        probe_and_check(build_keys, probe_keys, n, 3, 100);
        probe_and_check(build_keys, probe_keys, matches / 3, 2, 1024);
        std::printf("hash_probe n=%lld matches=%lld: ok\n", n, matches);
    }

    // Keys that all have the home slot capacity - 3, so the probes wrap around.
    const long long n = 1000;
    const long long capacity = table_capacity(n);
    std::vector<long long> keys(2 * n);
    for (long long i = 0; i < 2 * n; ++i) {
        keys[i] = fmix64_inverse(static_cast<unsigned long long>(i + 1) * capacity + (capacity - 3));
        check(keys[i] != -1, "colliding key is -1", n);
    }
    std::shuffle(keys.begin(), keys.end(), rng);
    const std::vector<long long> build_keys(keys.begin(), keys.begin() + n);
    std::shuffle(keys.begin(), keys.end(), rng);
    probe_and_check(build_keys, keys, 2 * n, default_grid(2 * n, block), block);
    probe_and_check(build_keys, keys, 2 * n, 1, 32);
    std::printf("hash_probe colliding n=%lld: ok\n", n);
    std::printf("PASS\n");
    return 0;
}
