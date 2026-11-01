// Test of the sorted_contains CUDA kernel against a CPU reference.
// Build: nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(const long long* keys, const long long* members, signed char* match, long long n, long long s);

#define CUDA_CHECK(x)                                                                        \
    do {                                                                                     \
        cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64()
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static long long random_value() { return (long long)(next_u64() % (1ull << 41)) - (1ll << 40); }

static bool run_case(long long n, long long s_target)
{
    std::vector<long long> members(std::max<long long>(s_target, 1));
    for (long long j = 0; j < s_target; ++j) members[j] = random_value();
    std::sort(members.begin(), members.begin() + s_target);
    members.erase(std::unique(members.begin(), members.begin() + s_target), members.begin() + s_target);
    const long long s = std::min<long long>(s_target, (long long)members.size());

    std::vector<long long> keys(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i)
        keys[i] = (s > 0 && (next_u64() & 1)) ? members[next_u64() % (uint64_t)s] : random_value();

    std::vector<signed char> expected(std::max<long long>(n, 1));
    for (long long i = 0; i < n; ++i)
        expected[i] = std::binary_search(members.begin(), members.begin() + s, keys[i]) ? 1 : 0;

    long long* d_keys = nullptr;
    long long* d_members = nullptr;
    signed char* d_match = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, std::max<long long>(n, 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_members, std::max<long long>(s, 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_match, std::max<long long>(n, 1)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    if (s > 0) CUDA_CHECK(cudaMemcpy(d_members, members.data(), s * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_match, 0xff, std::max<long long>(n, 1)));

    const unsigned blocks = (unsigned)std::max<long long>(1, (n + 255) / 256);
    kernel<<<blocks, 256>>>(d_keys, d_members, d_match, n, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<signed char> match(std::max<long long>(n, 1));
    if (n > 0) CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_match));

    for (long long i = 0; i < n; ++i) {
        if (match[i] != expected[i]) {
            printf("FAIL n=%lld s=%lld: match[%lld] = %d, expected %d\n", n, s, i, (int)match[i], (int)expected[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const long long sizes[] = {0, 1, 1000, 1000003};
    const long long member_counts[] = {0, 1, 100, 100000};
    bool ok = true;
    for (long long n : sizes) {
        for (long long s : member_counts) ok = run_case(n, s) && ok;
        printf("sorted_contains cuda n=%lld: %s\n", n, ok ? "ok" : "FAILED");
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
