// Test for hash_probe: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// The table is built on the CPU with the same hash and linear probing as
// hash_build. Half of the probe keys hit, half miss. The expected output is the
// set of (build row, probe row) pairs; the kernel may emit them in any order.
// A second run with out_cap below the match count checks that out_count still
// reports every match and that only valid pairs are written.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

extern "C" __global__ void kernel(const long long* probe_keys, const long long* table_keys,
                                  const int* table_rows, long long capacity, int* out_build,
                                  int* out_probe, int* out_count, long long m, long long out_cap);

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

static std::vector<long long> distinct_keys(long long n) {
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) keys[i] = i;
    for (long long i = n - 1; i > 0; --i) std::swap(keys[i], keys[next_u64() % (uint64_t)(i + 1)]);
    for (long long i = 0; i < n; ++i) keys[i] = keys[i] * 1000003LL + 17LL;
    return keys;
}

static void build_table(const std::vector<long long>& keys, long long capacity,
                        std::vector<long long>& table_keys, std::vector<int>& table_rows) {
    const uint64_t mask = (uint64_t)(capacity - 1);
    table_keys.assign(capacity, -1);
    table_rows.assign(capacity, -1);
    for (size_t i = 0; i < keys.size(); ++i) {
        uint64_t slot = mix64(keys[i]) & mask;
        while (table_keys[slot] != -1) slot = (slot + 1) & mask;
        table_keys[slot] = keys[i];
        table_rows[slot] = (int)i;
    }
}

static int lookup(const std::vector<long long>& table_keys, const std::vector<int>& table_rows,
                  long long capacity, long long key) {
    const uint64_t mask = (uint64_t)(capacity - 1);
    uint64_t slot = mix64(key) & mask;
    while (table_keys[slot] != -1) {
        if (table_keys[slot] == key) return table_rows[slot];
        slot = (slot + 1) & mask;
    }
    return -1;
}

static bool run_case(long long n, bool timed) {
    const long long m = n;
    const long long capacity = capacity_for(n);
    std::vector<long long> keys = distinct_keys(n);
    std::vector<long long> table_keys;
    std::vector<int> table_rows;
    build_table(keys, capacity, table_keys, table_rows);

    std::vector<long long> probe_keys(m);
    std::vector<std::pair<int, int>> expected;  // (probe row, build row)
    for (long long i = 0; i < m; ++i) {
        if (next_u64() & 1u) {
            probe_keys[i] = keys[next_u64() % (uint64_t)n];
            expected.push_back({(int)i, lookup(table_keys, table_rows, capacity, probe_keys[i])});
        } else {
            probe_keys[i] = -(i + 2);  // never in the table
        }
    }
    std::sort(expected.begin(), expected.end());
    const long long matches = (long long)expected.size();

    long long* d_probe_keys = nullptr;
    long long* d_table_keys = nullptr;
    int* d_table_rows = nullptr;
    int* d_out_build = nullptr;
    int* d_out_probe = nullptr;
    int* d_out_count = nullptr;
    const long long out_cap = std::max<long long>(matches, 1);
    CHECK(cudaMalloc(&d_probe_keys, std::max<long long>(m, 1) * sizeof(long long)));
    CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(long long)));
    CHECK(cudaMalloc(&d_table_rows, capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_out_build, out_cap * sizeof(int)));
    CHECK(cudaMalloc(&d_out_probe, out_cap * sizeof(int)));
    CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CHECK(cudaMemcpy(d_probe_keys, probe_keys.data(), m * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table_keys, table_keys.data(), capacity * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table_rows, table_rows.data(), capacity * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    kernel<<<grid_for(m), THREADS>>>(d_probe_keys, d_table_keys, d_table_rows, capacity, d_out_build,
                                     d_out_probe, d_out_count, m, out_cap);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int count = 0;
    CHECK(cudaMemcpy(&count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = (count == matches);
    if (ok) {
        std::vector<int> ob(count), op(count);
        CHECK(cudaMemcpy(ob.data(), d_out_build, count * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(op.data(), d_out_probe, count * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<std::pair<int, int>> got(count);
        for (int i = 0; i < count; ++i) got[i] = {op[i], ob[i]};
        std::sort(got.begin(), got.end());
        ok = (got == expected);
    }
    printf("  n=%lld: %s (matches=%d expected=%lld)\n", n, ok ? "ok" : "MISMATCH", count, matches);

    if (ok && matches >= 2) {
        // Output smaller than the match count: every match is counted, only valid pairs are written.
        const long long small_cap = matches / 2;
        CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
        CHECK(cudaMemset(d_out_build, 0xff, out_cap * sizeof(int)));
        CHECK(cudaMemset(d_out_probe, 0xff, out_cap * sizeof(int)));
        kernel<<<grid_for(m), THREADS>>>(d_probe_keys, d_table_keys, d_table_rows, capacity, d_out_build,
                                         d_out_probe, d_out_count, m, small_cap);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> ob(out_cap), op(out_cap);
        CHECK(cudaMemcpy(ob.data(), d_out_build, out_cap * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(op.data(), d_out_probe, out_cap * sizeof(int), cudaMemcpyDeviceToHost));
        bool small_ok = (count == matches);
        std::vector<char> seen(m, 0);
        for (long long i = 0; i < small_cap && small_ok; ++i) {
            small_ok = ob[i] >= 0 && ob[i] < n && op[i] >= 0 && op[i] < m && !seen[op[i]] &&
                       keys[ob[i]] == probe_keys[op[i]];
            if (small_ok) seen[op[i]] = 1;
        }
        for (long long i = small_cap; i < out_cap && small_ok; ++i) small_ok = (ob[i] == -1 && op[i] == -1);
        printf("  n=%lld out_cap=%lld: %s (out_count=%d)\n", n, small_cap, small_ok ? "ok" : "MISMATCH", count);
        ok = ok && small_ok;
    }

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(m), THREADS>>>(d_probe_keys, d_table_keys, d_table_rows, capacity,
                                             d_out_build, d_out_probe, d_out_count, m, out_cap);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_probe_keys));
    CHECK(cudaFree(d_table_keys));
    CHECK(cudaFree(d_table_rows));
    CHECK(cudaFree(d_out_build));
    CHECK(cudaFree(d_out_probe));
    CHECK(cudaFree(d_out_count));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s hash_probe\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
