// Test for groupby_sum: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// Two group cardinalities per size exercise both kernel paths: 1000 groups
// (capacity 2048, block-private shared-memory tables) and 100000 groups
// (capacity 262144, global table only). The check compares the set of keys and
// the per-key sums, and walks every key's probe chain.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int* group, const double* value, int* table_keys,
                                  double* table_sums, long long capacity, long long n);

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
static double next_double() { return (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

static uint64_t home_slot(int group, long long capacity) {
    uint64_t h = (uint64_t)(long long)group;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h & (uint64_t)(capacity - 1);
}

static const int THREADS = 256;
static const int SMEM_CAP = 2048;
static int grid_for(long long n, long long capacity) {
    const long long max_blocks = capacity <= SMEM_CAP ? 256 : 2048;
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > max_blocks) b = max_blocks;
    return (int)b;
}

static long long capacity_for(long long groups) {
    long long c = 2;
    while (c < 2 * groups) c *= 2;
    return c;
}

static bool validate(const std::vector<int>& group, const std::vector<double>& value,
                     const std::vector<int>& table_keys, const std::vector<double>& table_sums,
                     long long capacity, long long cardinality) {
    std::vector<double> ref(cardinality, 0.0);
    std::vector<char> present(cardinality, 0);
    long long distinct = 0;
    for (size_t i = 0; i < group.size(); ++i) {
        ref[group[i]] += value[i];
        if (!present[group[i]]) {
            present[group[i]] = 1;
            ++distinct;
        }
    }
    std::vector<char> seen(cardinality, 0);
    long long filled = 0;
    for (long long s = 0; s < capacity; ++s) {
        const int g = table_keys[s];
        if (g == -1) {
            if (table_sums[s] != 0.0) return false;
            continue;
        }
        if (g < 0 || g >= cardinality || !present[g] || seen[g]) return false;
        if (std::fabs(table_sums[s] - ref[g]) > 1e-9 * std::fabs(ref[g]) + 1e-9) return false;
        seen[g] = 1;
        ++filled;
    }
    if (filled != distinct) return false;
    const uint64_t mask = (uint64_t)(capacity - 1);
    for (long long g = 0; g < cardinality; ++g) {
        if (!present[g]) continue;
        uint64_t slot = home_slot((int)g, capacity);
        long long steps = 0;
        while (table_keys[slot] != g) {
            if (table_keys[slot] == -1 || ++steps > capacity) return false;
            slot = (slot + 1) & mask;
        }
    }
    return true;
}

static bool run_case(long long n, long long cardinality, bool timed) {
    const long long capacity = capacity_for(cardinality);
    std::vector<int> group(n);
    std::vector<double> value(n);
    for (long long i = 0; i < n; ++i) {
        group[i] = (int)(next_u64() % (uint64_t)cardinality);
        value[i] = next_double();
    }

    int* d_group = nullptr;
    double* d_value = nullptr;
    int* d_table_keys = nullptr;
    double* d_table_sums = nullptr;
    CHECK(cudaMalloc(&d_group, std::max<long long>(n, 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_value, std::max<long long>(n, 1) * sizeof(double)));
    CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_table_sums, capacity * sizeof(double)));
    CHECK(cudaMemcpy(d_group, group.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_value, value.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(int)));
    CHECK(cudaMemset(d_table_sums, 0, capacity * sizeof(double)));

    kernel<<<grid_for(n, capacity), THREADS>>>(d_group, d_value, d_table_keys, d_table_sums, capacity, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> table_keys(capacity);
    std::vector<double> table_sums(capacity);
    CHECK(cudaMemcpy(table_keys.data(), d_table_keys, capacity * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(table_sums.data(), d_table_sums, capacity * sizeof(double), cudaMemcpyDeviceToHost));
    const bool ok = validate(group, value, table_keys, table_sums, capacity, cardinality);
    printf("  n=%lld groups=%lld capacity=%lld: %s\n", n, cardinality, capacity, ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(int)));
            CHECK(cudaMemset(d_table_sums, 0, capacity * sizeof(double)));
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n, capacity), THREADS>>>(d_group, d_value, d_table_keys, d_table_sums, capacity, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld groups=%lld: %.1f us per launch\n", n, cardinality, total / reps * 1e3f);
    }

    CHECK(cudaFree(d_group));
    CHECK(cudaFree(d_value));
    CHECK(cudaFree(d_table_keys));
    CHECK(cudaFree(d_table_sums));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long cardinalities[] = {1000, 100000};
    bool ok = true;
    for (long long n : sizes)
        for (long long c : cardinalities) ok = run_case(n, c, n == 1000003) && ok;
    printf("%s groupby_sum\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
