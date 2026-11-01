#include <cstdint>

static constexpr unsigned long long EMPTY_KEY = 0xffffffffffffffffULL;

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n_elements,
    int64_t num_groups,
    unsigned long long* __restrict__ table,
    int64_t table_capacity,
    int64_t* __restrict__ counts
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }

    int64_t gid64 = group_ids[idx];
    int64_t supp64 = supp_ids[idx];

    if (gid64 < 0 || gid64 >= num_groups) {
        return;
    }

    unsigned long long gid = static_cast<unsigned long long>(static_cast<uint32_t>(gid64));
    unsigned long long supp = static_cast<unsigned long long>(static_cast<uint32_t>(supp64));
    unsigned long long key = (gid << 32) | supp;

    if (key == EMPTY_KEY) {
        key = 0xfffffffffffffffeULL;
    }

    unsigned long long mask = static_cast<unsigned long long>(table_capacity - 1);
    unsigned long long slot = mix64(key) & mask;

    for (int64_t probe = 0; probe < table_capacity; ++probe) {
        unsigned long long old = atomicCAS(&table[slot], EMPTY_KEY, key);
        if (old == EMPTY_KEY) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts[gid64]), 1ULL);
            return;
        }
        if (old == key) {
            return;
        }
        slot = (slot + 1ULL) & mask;
    }
}
