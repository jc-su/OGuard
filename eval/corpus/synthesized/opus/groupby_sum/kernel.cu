// Group-by sum: hash aggregation of value per group.
//
// The table uses open addressing with linear probing. The home slot of a group
// is fmix32(group) & (capacity - 1). Slot s holds a group in table_keys[s] and
// the sum of its values in table_sums[s].
//
// Each warp first adds up the values of the lanes that hold the same group, so
// only one lane per distinct group in the warp touches the table. This keeps
// the number of atomic adds low when there are few groups.
//
// Preconditions: the launch uses 1-D blocks, no group is -1, capacity is a
// power of two and at least 2 times the number of groups, every table_keys
// slot holds -1 and every table_sums slot holds 0 on entry.

constexpr int kEmpty = -1;

__device__ __forceinline__ unsigned hash_group(int group)
{
    // MurmurHash3 fmix32 finalizer.
    unsigned h = static_cast<unsigned>(group);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Mask of the lanes that exist in the calling warp of a 1-D block.
__device__ __forceinline__ unsigned warp_lanes_mask()
{
    const unsigned lane = threadIdx.x % 32;
    const unsigned lanes = min(32u, blockDim.x - (threadIdx.x - lane));
    return static_cast<unsigned>((1ULL << lanes) - 1);
}

extern "C" __global__ void kernel(const int* __restrict__ group,
                                  const double* __restrict__ value,
                                  int* __restrict__ table_keys,
                                  double* __restrict__ table_sums,
                                  long long capacity,
                                  long long n)
{
    const unsigned lane = threadIdx.x % 32;
    const unsigned lanes_mask = warp_lanes_mask();
    const unsigned lanes_below = (1u << lane) - 1;
    const unsigned lanes_above = 0xfffffffeu << lane;
    const unsigned long long mask = static_cast<unsigned long long>(capacity - 1);
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    // All lanes of a warp run the same number of iterations, so every lane
    // takes part in the warp vote.
    for (long long warp_start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x - lane;
         warp_start < n; warp_start += stride) {
        const long long j = warp_start + lane;
        const bool valid = j < n;
        const unsigned active = __ballot_sync(lanes_mask, valid);
        if (!valid) {
            continue;
        }
        const int g = group[j];
        double total = value[j];
        // Tree reduction over the lanes that hold the same group. In round r,
        // a lane whose rank among its peers has bit r set gives its partial sum
        // to the next lower remaining peer and stops.
        const unsigned peers = __match_any_sync(active, g);
        unsigned rank = __popc(peers & lanes_below);
        unsigned remaining = peers & lanes_above;
        while (__any_sync(active, remaining != 0)) {
            const int source = remaining != 0 ? __ffs(remaining) - 1 : static_cast<int>(lane);
            const double partial = __shfl_sync(active, total, source);
            if (remaining != 0) {
                total += partial;
            }
            remaining &= ~__ballot_sync(active, (rank & 1) != 0);
            rank >>= 1;
        }
        if ((peers & lanes_below) != 0) {
            continue;
        }
        unsigned long long slot = hash_group(g) & mask;
        while (true) {
            // A slot changes only from kEmpty to a group, so a stale read of a
            // group is never wrong; the CAS settles a stale read of kEmpty.
            int slot_key = table_keys[slot];
            if (slot_key == kEmpty) {
                slot_key = atomicCAS(table_keys + slot, kEmpty, g);
            }
            if (slot_key == kEmpty || slot_key == g) {
                atomicAdd(table_sums + slot, total);
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
}
