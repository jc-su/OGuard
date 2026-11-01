// Standalone test for hash_probe. Builds a hash table with a CPU
// reference implementation of hash_build, launches the hash_probe
// kernel, and checks the result against a CPU reference probe, for
// m in {0, 1, 1000, 1000003}.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

extern "C" __global__ void kernel(const long long *probe_keys,
                                   const long long *table_keys,
                                   const int *table_rows, int *out_build,
                                   int *out_probe, int *out_count,
                                   long long capacity, long long m,
                                   long long out_cap);

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

static unsigned long long cpu_hash64(long long key) {
    unsigned long long h = (unsigned long long)key;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h = h ^ (h >> 31);
    return h;
}

static void cpu_build(const std::vector<long long> &keys, long long capacity,
                       std::vector<long long> &table_keys,
                       std::vector<int> &table_rows) {
    table_keys.assign(capacity, -1);
    table_rows.assign(capacity, 0);
    unsigned long long mask = (unsigned long long)(capacity - 1);
    for (size_t i = 0; i < keys.size(); ++i) {
        long long slot = (long long)(cpu_hash64(keys[i]) & mask);
        while (table_keys[slot] != -1) {
            slot = (long long)(((unsigned long long)slot + 1) & mask);
        }
        table_keys[slot] = keys[i];
        table_rows[slot] = (int)i;
    }
}

static void cpu_probe(const std::vector<long long> &probe_keys,
                       const std::vector<long long> &table_keys,
                       const std::vector<int> &table_rows, long long capacity,
                       std::vector<std::pair<int, int>> &pairs) {
    unsigned long long mask = (unsigned long long)(capacity - 1);
    for (size_t j = 0; j < probe_keys.size(); ++j) {
        long long key = probe_keys[j];
        long long slot = (long long)(cpu_hash64(key) & mask);
        while (table_keys[slot] != -1) {
            if (table_keys[slot] == key) {
                pairs.push_back({(int)j, table_rows[slot]});
                break;
            }
            slot = (long long)(((unsigned long long)slot + 1) & mask);
        }
    }
}

static bool run_case(long long m, std::mt19937_64 &rng,
                      const std::vector<long long> &build_keys,
                      const std::vector<long long> &table_keys,
                      const std::vector<int> &table_rows, long long capacity) {
    long long build_n = (long long)build_keys.size();

    std::vector<long long> probe_keys(m > 0 ? m : 0);
    if (m > 0) {
        std::uniform_int_distribution<long long> pick_build(0, build_n - 1);
        std::uniform_int_distribution<long long> pick_absent(-(4 * build_n) - 4,
                                                              -4);
        for (long long j = 0; j < m; ++j) {
            probe_keys[j] =
                (j % 2 == 0) ? build_keys[pick_build(rng)] : pick_absent(rng);
        }
        std::shuffle(probe_keys.begin(), probe_keys.end(), rng);
    }

    std::vector<std::pair<int, int>> ref_pairs;
    cpu_probe(probe_keys, table_keys, table_rows, capacity, ref_pairs);

    long long out_cap = m > 0 ? m : 1;

    long long *d_probe_keys, *d_table_keys;
    int *d_table_rows, *d_out_build, *d_out_probe, *d_out_count;
    CUDA_CHECK(cudaMalloc(&d_probe_keys, sizeof(long long) * (m > 0 ? m : 1)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, sizeof(long long) * capacity));
    CUDA_CHECK(cudaMalloc(&d_table_rows, sizeof(int) * capacity));
    CUDA_CHECK(cudaMalloc(&d_out_build, sizeof(int) * out_cap));
    CUDA_CHECK(cudaMalloc(&d_out_probe, sizeof(int) * out_cap));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    if (m > 0) {
        CUDA_CHECK(cudaMemcpy(d_probe_keys, probe_keys.data(),
                              sizeof(long long) * m, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_table_keys, table_keys.data(),
                          sizeof(long long) * capacity, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_rows, table_rows.data(), sizeof(int) * capacity,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    long long blocks = (m + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_probe_keys, d_table_keys, d_table_rows,
                                   d_out_build, d_out_probe, d_out_count,
                                   capacity, m, out_cap);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int got_count = 0;
    CUDA_CHECK(cudaMemcpy(&got_count, d_out_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool ok = (got_count == (int)ref_pairs.size());
    if (ok && got_count > 0) {
        std::vector<int> got_build(got_count), got_probe(got_count);
        CUDA_CHECK(cudaMemcpy(got_build.data(), d_out_build,
                              sizeof(int) * got_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_probe.data(), d_out_probe,
                              sizeof(int) * got_count, cudaMemcpyDeviceToHost));
        std::vector<std::pair<int, int>> got_pairs(got_count);
        for (int i = 0; i < got_count; ++i)
            got_pairs[i] = {got_probe[i], got_build[i]};
        std::sort(got_pairs.begin(), got_pairs.end());
        std::sort(ref_pairs.begin(), ref_pairs.end());
        ok = (got_pairs == ref_pairs);
    }

    printf("m=%lld: %s (matches=%d, ref=%zu)\n", m, ok ? "PASS" : "FAIL",
           got_count, ref_pairs.size());

    cudaFree(d_probe_keys);
    cudaFree(d_table_keys);
    cudaFree(d_table_rows);
    cudaFree(d_out_build);
    cudaFree(d_out_probe);
    cudaFree(d_out_count);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);

    long long build_n = 5000;
    long long capacity = std::max(2LL, next_pow2(2 * build_n));

    std::vector<long long> pool(4 * build_n + 4);
    std::iota(pool.begin(), pool.end(), 1);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<long long> build_keys(pool.begin(), pool.begin() + build_n);

    std::vector<long long> table_keys;
    std::vector<int> table_rows;
    cpu_build(build_keys, capacity, table_keys, table_rows);

    bool all_ok = true;
    for (long long m : {0LL, 1LL, 1000LL, 1000003LL}) {
        all_ok &= run_case(m, rng, build_keys, table_keys, table_rows, capacity);
    }
    return all_ok ? 0 : 1;
}
