// GPU test of kernel.cu against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_hash_build

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
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

static void build_and_check(const std::vector<long long>& keys, long long capacity, int grid,
                            int block)
{
    const long long n = static_cast<long long>(keys.size());
    long long* d_keys = nullptr;
    long long* d_table_keys = nullptr;
    int* d_table_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_table_keys, capacity * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_table_rows, capacity * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table_keys, 0xff, capacity * sizeof(long long)));  // all -1
    CUDA_CHECK(cudaMemset(d_table_rows, 0, capacity * sizeof(int)));

    kernel<<<grid, block>>>(d_keys, d_table_keys, d_table_rows, capacity, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<long long> table_keys(capacity);
    std::vector<int> table_rows(capacity);
    CUDA_CHECK(cudaMemcpy(table_keys.data(), d_table_keys, capacity * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(table_rows.data(), d_table_rows, capacity * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_table_keys));
    CUDA_CHECK(cudaFree(d_table_rows));

    const long long occupied =
        std::count_if(table_keys.begin(), table_keys.end(), [](long long k) { return k != -1; });
    check(occupied == n, "occupied slot count differs from n", n);
    // Distinct keys land in distinct slots, so finding every key by a linear
    // probe from its home slot proves the table holds exactly these keys.
    const unsigned long long mask = static_cast<unsigned long long>(capacity - 1);
    for (long long i = 0; i < n; ++i) {
        unsigned long long slot = fmix64(static_cast<unsigned long long>(keys[i])) & mask;
        while (table_keys[slot] != keys[i]) {
            check(table_keys[slot] != -1, "key not reachable by a linear probe", n);
            slot = (slot + 1) & mask;
        }
        check(table_rows[slot] == i, "slot row id differs from the key's row", n);
    }
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const int block = 256;
    std::mt19937_64 rng(5);
    for (const long long n : sizes) {
        // Distinct keys: an odd multiplier is a bijection on 64-bit integers.
        std::vector<long long> keys(n);
        for (long long i = 0; i < n; ++i) {
            keys[i] = static_cast<long long>((static_cast<unsigned long long>(i) + 12345) *
                                             0x9e3779b97f4a7c15ULL);
        }
        std::shuffle(keys.begin(), keys.end(), rng);
        const long long capacity = table_capacity(n);
        build_and_check(keys, capacity, default_grid(n, block), block);
        build_and_check(keys, capacity, 3, 96);

        std::vector<long long> sequential;
        for (long long i = 0; i < n; ++i) {
            if (i - n / 2 != -1) {
                sequential.push_back(i - n / 2);
            }
        }
        build_and_check(sequential, 4 * capacity, default_grid(n, block), block);
        std::printf("hash_build n=%lld: ok\n", n);
    }

    // Keys that all have the home slot capacity - 3, so the probes wrap around.
    const long long n = 1000;
    const long long capacity = table_capacity(n);
    std::vector<long long> colliding(n);
    for (long long i = 0; i < n; ++i) {
        colliding[i] = fmix64_inverse(static_cast<unsigned long long>(i + 1) * capacity +
                                      (capacity - 3));
        check(colliding[i] != -1, "colliding key is -1", n);
    }
    std::shuffle(colliding.begin(), colliding.end(), rng);
    build_and_check(colliding, capacity, default_grid(n, block), block);
    std::printf("hash_build colliding n=%lld: ok\n", n);
    std::printf("PASS\n");
    return 0;
}
