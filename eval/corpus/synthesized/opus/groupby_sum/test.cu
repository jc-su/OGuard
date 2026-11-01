// GPU test of kernel.cu against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_groupby_sum

#include <algorithm>
#include <cmath>
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

static void check(bool ok, const char* what, long long n)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL n=%lld: %s\n", n, what);
        std::exit(1);
    }
}

static unsigned fmix32(unsigned h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static unsigned inverse_odd(unsigned c)
{
    unsigned x = c;  // Newton iteration doubles the correct low bits.
    for (int i = 0; i < 5; ++i) {
        x *= 2 - c * x;
    }
    return x;
}

static int fmix32_inverse(unsigned x)
{
    x ^= x >> 16;
    x *= inverse_odd(0xc2b2ae35u);
    x ^= (x >> 13) ^ (x >> 26);
    x *= inverse_odd(0x85ebca6bu);
    x ^= x >> 16;
    return static_cast<int>(x);
}

static long long table_capacity(long long groups)
{
    long long capacity = 1;
    while (capacity < 2 * groups) {
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

static void aggregate_and_check(const std::vector<int>& group, const std::vector<double>& value,
                                long long capacity, int grid, int block)
{
    const long long n = static_cast<long long>(group.size());
    int* d_group = nullptr;
    double* d_value = nullptr;
    int* d_table_keys = nullptr;
    double* d_table_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_group, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_value, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table_sums, capacity * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_group, group.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, value.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(int)));  // all -1
    CUDA_CHECK(cudaMemset(d_table_sums, 0, capacity * sizeof(double)));

    kernel<<<grid, block>>>(d_group, d_value, d_table_keys, d_table_sums, capacity, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> table_keys(capacity);
    std::vector<double> table_sums(capacity);
    CUDA_CHECK(cudaMemcpy(table_keys.data(), d_table_keys, capacity * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(table_sums.data(), d_table_sums, capacity * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_group));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_table_keys));
    CUDA_CHECK(cudaFree(d_table_sums));

    std::unordered_map<int, double> want_sum;
    std::unordered_map<int, double> scale;
    for (long long i = 0; i < n; ++i) {
        want_sum[group[i]] += value[i];
        scale[group[i]] += std::fabs(value[i]);
    }
    long long occupied = 0;
    for (long long s = 0; s < capacity; ++s) {
        if (table_keys[s] == -1) {
            check(table_sums[s] == 0.0, "sum in an empty slot", n);
            continue;
        }
        ++occupied;
        const auto it = want_sum.find(table_keys[s]);
        check(it != want_sum.end(), "group not in the input", n);
        const double error = std::fabs(table_sums[s] - it->second);
        check(error <= 1e-9 * scale[table_keys[s]] + 1e-300, "wrong sum", n);
    }
    // Every occupied slot holds an input group, so equal counts mean every
    // group appears exactly once.
    check(occupied == static_cast<long long>(want_sum.size()), "group count differs", n);
}

// Distinct groups, negative ones included, none equal to -1. An odd
// multiplier is a bijection on 31-bit integers.
static std::vector<int> distinct_groups(long long count, std::mt19937_64& rng)
{
    std::vector<int> groups(count);
    for (long long i = 0; i < count; ++i) {
        const long long g = (((i + 777) * 0x2545f491LL) & 0x7fffffffLL) - (1LL << 30);
        groups[i] = static_cast<int>(g == -1 ? (1LL << 30) : g);
    }
    std::shuffle(groups.begin(), groups.end(), rng);
    return groups;
}

static void run_cases(const std::vector<int>& keys, long long n, long long capacity,
                      std::mt19937_64& rng)
{
    const int block = 256;
    std::normal_distribution<double> normal(0.0, 100.0);
    std::vector<int> group(n);
    std::vector<double> value(n);
    for (long long i = 0; i < n; ++i) {
        group[i] = keys[rng() % keys.size()];
        value[i] = normal(rng);
    }
    aggregate_and_check(group, value, capacity, default_grid(n, block), block);
    aggregate_and_check(group, value, capacity, 3, 100);
    aggregate_and_check(group, value, 4 * capacity, 2, 1024);
    aggregate_and_check(group, value, capacity, 1, 32);
    std::sort(group.begin(), group.end());
    aggregate_and_check(group, value, capacity, default_grid(n, block), block);
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    std::mt19937_64 rng(7);
    for (const long long n : sizes) {
        if (n == 0) {
            aggregate_and_check({}, {}, 1, default_grid(0, 256), 256);
            std::printf("groupby_sum n=0: ok\n");
            continue;
        }
        std::vector<long long> group_counts = {1, std::min(n, 1000LL), std::max(1LL, n / 2)};
        std::sort(group_counts.begin(), group_counts.end());
        group_counts.erase(std::unique(group_counts.begin(), group_counts.end()),
                           group_counts.end());
        for (const long long groups : group_counts) {
            run_cases(distinct_groups(groups, rng), n, table_capacity(groups), rng);
            std::printf("groupby_sum n=%lld groups=%lld: ok\n", n, groups);
        }
    }

    // Groups that all have the home slot capacity - 2, so the probes wrap around.
    const long long groups = 100;
    const long long capacity = table_capacity(groups);
    std::vector<int> colliding(groups);
    for (long long i = 0; i < groups; ++i) {
        colliding[i] = fmix32_inverse(static_cast<unsigned>((i + 1) * capacity + capacity - 2));
        check(colliding[i] != -1, "colliding group is -1", groups);
        check((fmix32(static_cast<unsigned>(colliding[i])) & (capacity - 1)) ==
                  static_cast<unsigned>(capacity - 2),
              "colliding group has another home slot", groups);
    }
    for (const long long n : {1000LL, 1000003LL}) {
        run_cases(colliding, n, capacity, rng);
        std::printf("groupby_sum colliding n=%lld groups=%lld: ok\n", n, groups);
    }
    std::printf("PASS\n");
    return 0;
}
