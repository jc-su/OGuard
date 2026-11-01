#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group,
    const int64_t* __restrict__ supp,
    int64_t n,
    unsigned long long* __restrict__ out)          // use 64‑bit unsigned for atomicAdd
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool is_unique = false;
    if (idx == 0) {
        is_unique = true;
    } else {
        int64_t g0 = group[idx];
        int64_t s0 = supp[idx];
        int64_t g1 = group[idx - 1];
        int64_t s1 = supp[idx - 1];
        if (g0 != g1 || s0 != s1) {
            is_unique = true;
        }
    }

    if (is_unique) {
        // one atomic per distinct pair
        atomicAdd(&out[group[idx]], 1ULL);
    }
}
