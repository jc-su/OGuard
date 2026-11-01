// Test for hash_build: launches kernel.cu and validates the table on the CPU.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// The GPU places colliding keys in an order that depends on scheduling, so the
// check validates the table's properties instead of comparing slot by slot:
// every row appears exactly once, each slot is consistent, and a linear probe
// from every key's home slot reaches the key without crossing an empty slot.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const long long* keys, long long* table_keys, int* table_rows,
                                  long long capacity, long long n);

#define CHECK(call)                                                                    \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint64_t mix64(long long key) {
    uint64_t h = (uint64_t)key;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
}

static const int THREADS = 256;
static const int MAX_BLOCKS = 2048;
static int grid_for(long long n) {
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return (int)b;
}

static long long capacity_for(long long n) {
    long long c = 2;
    while (c < 2 * n) c *= 2;
    return c;
}

// Distinct positive keys: a random permutation of 0..n-1 times an odd constant.
static std::vector<long long> distinct_keys(long long n) {
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) keys[i] = i;
    for (long long i = n - 1; i > 0; --i) std::swap(keys[i], keys[next_u64() % (uint64_t)(i + 1)]);
    for (long long i = 0; i < n; ++i) keys[i] = keys[i] * 1000003LL + 17LL;
    return keys;
}

static bool validate(const std::vector<long long>& keys, const std::vector<long long>& table_keys,
                     const std::vector<int>& table_rows, long long capacity) {
    const long long n = (long long)keys.size();
    const uint64_t mask = (uint64_t)(capacity - 1);
    std::vector<char> seen(n, 0);
    long long filled = 0;
    for (long long s = 0; s < capacity; ++s) {
        if (table_keys[s] == -1) continue;
        ++filled;
        const int row = table_rows[s];
        if (row < 0 || row >= n || seen[row] || keys[row] != table_keys[s]) return false;
        seen[row] = 1;
    }
    if (filled != n) return false;
    for (long long i = 0; i < n; ++i) {
        uint64_t slot = mix64(keys[i]) & mask;
        long long steps = 0;
        while (table_keys[slot] != keys[i]) {
            if (table_keys[slot] == -1 || ++steps > capacity) return false;
            slot = (slot + 1) & mask;
        }
    }
    return true;
}

static bool run_case(long long n, bool timed) {
    const long long capacity = capacity_for(n);
    std::vector<long long> keys = distinct_keys(n);

    long long* d_keys = nullptr;
    long long* d_table_keys = nullptr;
    int* d_table_rows = nullptr;
    CHECK(cudaMalloc(&d_keys, std::max<long long>(n, 1) * sizeof(long long)));
    CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(long long)));
    CHECK(cudaMalloc(&d_table_rows, capacity * sizeof(int)));
    CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(long long)));

    kernel<<<grid_for(n), THREADS>>>(d_keys, d_table_keys, d_table_rows, capacity, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<long long> table_keys(capacity);
    std::vector<int> table_rows(capacity);
    CHECK(cudaMemcpy(table_keys.data(), d_table_keys, capacity * sizeof(long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(table_rows.data(), d_table_rows, capacity * sizeof(int), cudaMemcpyDeviceToHost));
    const bool ok = validate(keys, table_keys, table_rows, capacity);
    printf("  n=%lld capacity=%lld: %s\n", n, capacity, ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(long long)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_keys, d_table_keys, d_table_rows, capacity, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_table_keys));
    CHECK(cudaFree(d_table_rows));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s hash_build\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
