// Standalone test for groupby_sum. Allocates inputs, launches the
// kernel, and checks the resulting table against a CPU reference sum
// per group, for n in {0, 1, 1000, 1000003}.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

extern "C" __global__ void kernel(const int *group, const double *value,
                                   int *table_keys, double *table_sums,
                                   long long capacity, long long n);

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

const int BLOCK = 256;

static long long next_pow2(long long x) {
    long long p = 1;
    while (p < x) p *= 2;
    return p;
}

static bool run_case(long long n, std::mt19937_64 &rng) {
    long long num_groups = n > 0 ? std::min(n, 20000LL) : 1;
    long long capacity = std::max(2LL, next_pow2(2 * num_groups));

    std::vector<int> group(n > 0 ? n : 1);
    std::vector<double> value(n > 0 ? n : 1);
    std::uniform_int_distribution<int> group_dist(0, (int)num_groups - 1);
    std::uniform_real_distribution<double> value_dist(-10.0, 10.0);
    for (long long i = 0; i < n; ++i) {
        group[i] = group_dist(rng);
        value[i] = value_dist(rng);
    }

    std::map<int, double> ref_sums;
    for (long long i = 0; i < n; ++i) ref_sums[group[i]] += value[i];

    int *d_group, *d_table_keys;
    double *d_value, *d_table_sums;
    CUDA_CHECK(cudaMalloc(&d_group, sizeof(int) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_value, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, sizeof(int) * capacity));
    CUDA_CHECK(cudaMalloc(&d_table_sums, sizeof(double) * capacity));
    CUDA_CHECK(cudaMemcpy(d_group, group.data(), sizeof(int) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, value.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));

    std::vector<int> init_keys(capacity, -1);
    CUDA_CHECK(cudaMemcpy(d_table_keys, init_keys.data(), sizeof(int) * capacity,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table_sums, 0, sizeof(double) * capacity));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_group, d_value, d_table_keys, d_table_sums,
                                   capacity, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> table_keys(capacity);
    std::vector<double> table_sums(capacity);
    CUDA_CHECK(cudaMemcpy(table_keys.data(), d_table_keys, sizeof(int) * capacity,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(table_sums.data(), d_table_sums,
                          sizeof(double) * capacity, cudaMemcpyDeviceToHost));

    std::set<int> found_groups;
    bool ok = true;
    for (long long s = 0; s < capacity; ++s) {
        if (table_keys[s] == -1) continue;
        found_groups.insert(table_keys[s]);
        double ref = ref_sums[table_keys[s]];
        double tol = 1e-6 * std::max(1.0, std::fabs(ref));
        if (std::fabs(table_sums[s] - ref) > tol) ok = false;
    }
    if (found_groups.size() != ref_sums.size()) ok = false;

    printf("n=%lld: %s (groups=%zu, capacity=%lld)\n", n, ok ? "PASS" : "FAIL",
           found_groups.size(), capacity);

    cudaFree(d_group);
    cudaFree(d_value);
    cudaFree(d_table_keys);
    cudaFree(d_table_sums);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);
    bool all_ok = true;
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        all_ok &= run_case(n, rng);
    }
    return all_ok ? 0 : 1;
}
