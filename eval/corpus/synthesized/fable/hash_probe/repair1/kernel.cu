// hash_probe: emit (build row, probe row) for every probe key found in the table.
//
// Each lane probes one row with linear probing from mix64(key) & (capacity - 1)
// and stops at the first empty slot, so a build side with duplicate keys emits
// every match. Output writes are warp-aggregated: on every probe step the warp
// ballots its matches, lane 0 reserves the slots with one atomicAdd on out_count,
// and matching lanes write at base + rank. The probe loop runs in lockstep for
// the whole warp (the row loop is rounded up to a multiple of the warp size) so
// every ballot is a full-warp ballot.
//
// The output position is formed from base as an unsigned value and compared
// against out_cap before the store, so every store stays inside out_build and
// out_probe (out_cap slots) whatever out_count holds on entry. Pairs past
// out_cap are counted but not written, so the caller can size the output from
// out_count and probe again.
//
// out_count must be zero on entry.

static __device__ __forceinline__ unsigned long long mix64(long long key) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(const long long* __restrict__ probe_keys,
                                  const long long* __restrict__ table_keys,
                                  const int* __restrict__ table_rows,
                                  long long capacity,
                                  int* __restrict__ out_build,
                                  int* __restrict__ out_probe,
                                  int* __restrict__ out_count,
                                  long long m,
                                  long long out_cap) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const unsigned long long mask = (unsigned long long)(capacity - 1);
    const long long m_round = (m + 31) & ~31LL;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < m_round; i += stride) {
        long long key = 0;
        unsigned long long slot = 0;
        bool done = true;
        if (i < m) {
            key = probe_keys[i];
            slot = mix64(key) & mask;
            done = false;
        }
        while (true) {
            bool found = false;
            int build_row = 0;
            if (!done) {
                const long long tk = table_keys[slot];
                if (tk == -1LL) {
                    done = true;
                } else if (tk == key) {
                    found = true;
                    build_row = table_rows[slot];
                }
                slot = (slot + 1) & mask;
            }
            const unsigned ballot = __ballot_sync(0xffffffffu, found);
            if (ballot != 0u) {
                int base = 0;
                if (lane == 0u) base = atomicAdd(out_count, __popc(ballot));
                base = __shfl_sync(0xffffffffu, base, 0);
                if (found) {
                    const long long pos = (unsigned)base + (unsigned)__popc(ballot & lanemask_lt);
                    if (pos < out_cap) {
                        out_build[pos] = build_row;
                        out_probe[pos] = (int)i;
                    }
                }
            }
            if (__ballot_sync(0xffffffffu, !done) == 0u) break;
        }
    }
}
