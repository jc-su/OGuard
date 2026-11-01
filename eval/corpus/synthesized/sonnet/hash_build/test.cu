// Standalone test for hash_build. Allocates inputs, launches the
// kernel, and checks the resulting table against the invariants of a
// correct build, for n in {0, 1, 1000, 1000003}.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const long long *keys, long long *table_keys,
                                   int *table_rows, long long capacity,
                                   long long n);

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
    long long capacity = std::max(2LL, next_pow2(2 * n));

    // Distinct keys, drawn without replacement, never -1.
    std::vector<long long> pool(4 * n + 4);
    std::iota(pool.begin(), pool.end(), 1);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<long long> keys(pool.begin(), pool.begin() + n);

    long long *d_keys, *d_table_keys;
    int *d_table_rows;
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(long long) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, sizeof(long long) * capacity));
    CUDA_CHECK(cudaMalloc(&d_table_rows, sizeof(int) * capacity));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), sizeof(long long) * n,
                          cudaMemcpyHostToDevice));

    std::vector<long long> init_keys(capacity, -1);
    CUDA_CHECK(cudaMemcpy(d_table_keys, init_keys.data(),
                          sizeof(long long) * capacity,
                          cudaMemcpyHostToDevice));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_keys, d_table_keys, d_table_rows, capacity,
                                   n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> table_keys(capacity);
    std::vector<int> table_rows(capacity);
    CUDA_CHECK(cudaMemcpy(table_keys.data(), d_table_keys,
                          sizeof(long long) * capacity,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(table_rows.data(), d_table_rows, sizeof(int) * capacity,
                          cudaMemcpyDeviceToHost));

    long long occupied = 0;
    std::vector<char> row_seen(n > 0 ? n : 1, 0);
    bool ok = true;
    for (long long s = 0; s < capacity; ++s) {
        if (table_keys[s] == -1) continue;
        occupied++;
        int row = table_rows[s];
        if (row < 0 || row >= n || row_seen[row]) {
            ok = false;
            continue;
        }
        row_seen[row] = 1;
        if (table_keys[s] != keys[row]) ok = false;
    }
    if (occupied != n) ok = false;

    printf("n=%lld: %s (capacity=%lld, occupied=%lld)\n", n, ok ? "PASS" : "FAIL",
           capacity, occupied);

    cudaFree(d_keys);
    cudaFree(d_table_keys);
    cudaFree(d_table_rows);
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
