// Hash probe: look up each probe key in the table that hash_build wrote.
//
// The table uses open addressing with linear probing. The home slot of a key
// is fmix64(key) & (capacity - 1), the same as in hash_build. The build keys
// are distinct, so each probe row matches at most one build row.
//
// Each warp appends its matches with one atomic add on out_count: the lanes
// with a match take consecutive positions in their lane order. The kernel
// writes at most out_cap pairs; out_count receives the number of matches.
//
// Preconditions: the launch uses 1-D blocks, capacity is a power of two, the
// table has at least one empty slot (-1), and out_count holds 0 on entry.

constexpr long long kEmpty = -1;

__device__ __forceinline__ unsigned long long hash_key(long long key)
{
    // MurmurHash3 fmix64 finalizer.
    unsigned long long h = static_cast<unsigned long long>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// Mask of the lanes that exist in the calling warp of a 1-D block.
__device__ __forceinline__ unsigned warp_lanes_mask()
{
    const unsigned lane = threadIdx.x % 32;
    const unsigned lanes = min(32u, blockDim.x - (threadIdx.x - lane));
    return static_cast<unsigned>((1ULL << lanes) - 1);
}

extern "C" __global__ void kernel(const long long* __restrict__ probe_keys,
                                  const long long* __restrict__ table_keys,
                                  const int* __restrict__ table_rows,
                                  long long capacity,
                                  int* __restrict__ out_build,
                                  int* __restrict__ out_probe,
                                  int* __restrict__ out_count,
                                  long long m,
                                  long long out_cap)
{
    const unsigned lane = threadIdx.x % 32;
    const unsigned lanes_mask = warp_lanes_mask();
    const unsigned lanes_below = (1u << lane) - 1;
    const unsigned long long mask = static_cast<unsigned long long>(capacity - 1);
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    // All lanes of a warp run the same number of iterations, so every lane
    // takes part in the warp vote.
    for (long long warp_start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x - lane;
         warp_start < m; warp_start += stride) {
        const long long j = warp_start + lane;
        bool found = false;
        int build_row = 0;
        if (j < m) {
            const long long key = probe_keys[j];
            unsigned long long slot = hash_key(key) & mask;
            while (true) {
                const long long slot_key = table_keys[slot];
                if (slot_key == kEmpty) {
                    break;
                }
                if (slot_key == key) {
                    found = true;
                    build_row = table_rows[slot];
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
        const unsigned hits = __ballot_sync(lanes_mask, found);
        if (found) {
            const int rank = __popc(hits & lanes_below);
            const int leader = __ffs(hits) - 1;
            int first = 0;
            if (rank == 0) {
                first = atomicAdd(out_count, __popc(hits));
            }
            first = __shfl_sync(hits, first, leader);
            const long long position = static_cast<long long>(first) + rank;
            if (position < out_cap) {
                out_build[position] = build_row;
                out_probe[position] = static_cast<int>(j);
            }
        }
    }
}
