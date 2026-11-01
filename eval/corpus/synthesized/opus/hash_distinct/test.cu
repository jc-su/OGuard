#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernel.cu"

#define CHECK(x)                                                                     \
    do {                                                                             \
        cudaError_t err_ = (x);                                                      \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64()
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

template <class T> static T* to_device(const std::vector<T>& h)
{
    T* d = nullptr;
    CHECK(cudaMalloc(&d, (h.size() > 0 ? h.size() : 1) * sizeof(T)));
    if (!h.empty()) CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <class T> static std::vector<T> to_host(const T* d, size_t n)
{
    std::vector<T> h(n);
    if (n) CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

[[maybe_unused]] static int sm_count()
{
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    return sms;
}

// Blocks of 256 threads for n items, at most blocks_per_sm per SM, at least 1.
[[maybe_unused]] static int blocks_for(long long n, int blocks_per_sm)
{
    long long blocks = (n + 255) / 256;
    const long long cap = (long long)sm_count() * blocks_per_sm;
    if (blocks > cap) blocks = cap;
    return blocks < 1 ? 1 : (int)blocks;
}

#include <unordered_map>

static unsigned long long start_slot(long long key, long long capacity)
{
    return (((unsigned long long)key * 0x9E3779B97F4A7C15ull) >> 32) & (unsigned long long)(capacity - 1);
}

// mode 0: random keys, 1: many duplicates, 2: one key, 3: keys sharing start slots
static bool run_case(long long n, int mode, int spread)
{
    long long capacity = 1;
    while (capacity < 2 * n * spread) capacity <<= 1;
    std::vector<long long> keys(n);
    for (long long i = 0; i < n; ++i) {
        const uint64_t r = next_u64();
        switch (mode) {
        case 0: keys[i] = (long long)r; break;
        case 1: keys[i] = (long long)(r % (uint64_t)(n / 10 + 1)) - 5; break;
        case 2: keys[i] = -2; break;
        default: keys[i] = (long long)((r % (uint64_t)(n / 2 + 1)) << 32); break;
        }
        if (keys[i] == -1) keys[i] = 0;
    }
    std::vector<long long> tk_init(capacity, -1);
    std::vector<int> tr_init(capacity, 0x7fffffff);

    long long* d_keys = to_device(keys);
    long long* d_tk = to_device(tk_init);
    int* d_tr = to_device(tr_init);
    kernel<<<blocks_for(n, 32), 256>>>(d_keys, d_tk, d_tr, n, capacity);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> tk = to_host(d_tk, capacity);
    std::vector<int> tr = to_host(d_tr, capacity);
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_tk));
    CHECK(cudaFree(d_tr));

    std::unordered_map<long long, long long> first_row;
    for (long long i = 0; i < n; ++i) first_row.emplace(keys[i], i);
    long long occupied = 0;
    for (long long s = 0; s < capacity; ++s) {
        if (tk[s] == -1) {
            if (tr[s] != 0x7fffffff) {
                printf("FAIL n=%lld mode=%d: empty slot %lld has row %d\n", n, mode, s, tr[s]);
                return false;
            }
            continue;
        }
        ++occupied;
        auto it = first_row.find(tk[s]);
        if (it == first_row.end() || tr[s] != it->second) {
            printf("FAIL n=%lld mode=%d: slot %lld key %lld row %d\n", n, mode, s, tk[s], tr[s]);
            return false;
        }
        for (unsigned long long p = start_slot(tk[s], capacity); p != (unsigned long long)s;
             p = (p + 1) & (capacity - 1)) {
            if (tk[p] == -1) {
                printf("FAIL n=%lld mode=%d: key %lld in slot %lld passes empty slot %llu\n", n, mode, tk[s], s, p);
                return false;
            }
        }
    }
    if (occupied != (long long)first_row.size()) {
        printf("FAIL n=%lld mode=%d: %lld occupied slots for %zu distinct keys\n", n, mode, occupied, first_row.size());
        return false;
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) {
        for (int mode = 0; mode < 4; ++mode) {
            ok = run_case(n, mode, 1) && ok;
            ok = run_case(n, mode, 4) && ok;
        }
        printf("n=%lld %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
