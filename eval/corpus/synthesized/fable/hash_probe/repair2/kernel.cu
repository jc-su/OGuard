// hash_probe: emit (build row, probe row) for every probe key found in the table.
//
// Each thread probes one row at a time with linear probing from
// mix64(key) & (capacity - 1) and stops at the first empty slot, so a build
// side with duplicate keys emits every match. Output writes are
// warp-aggregated: on a match, the thread reads the active mask, the lowest
// active lane reserves the slots for all active lanes with one atomicAdd on
// out_count, and every active lane writes at base + (number of active lanes
// below it). Under independent thread scheduling the active mask can cover a
// subset of the lanes that hold a match; each subset then reserves its own
// slots, which is still correct.
//
// The store sits inside the row loop, whose condition i < m bounds the stored
// probe row id on every path to the store. The output position is formed from
// base as an unsigned value and compared against out_cap before the store, so
// every store stays inside out_build and out_probe (out_cap slots) whatever
// out_count holds on entry. Pairs past out_cap are counted but not written, so
// the caller can size the output from out_count and probe again.
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
    const unsigned lanemask_lt = (1u << (threadIdx.x & 31u)) - 1u;
    const unsigned long long mask = (unsigned long long)(capacity - 1);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        const long long key = probe_keys[i];
        for (unsigned long long slot = mix64(key) & mask;; slot = (slot + 1) & mask) {
            const long long tk = table_keys[slot];
            if (tk == -1LL) break;
            if (tk != key) continue;
            const int build_row = table_rows[slot];
            const unsigned active = __activemask();
            const unsigned rank = __popc(active & lanemask_lt);
            int base = 0;
            if (rank == 0u) base = atomicAdd(out_count, __popc(active));
            base = __shfl_sync(active, base, __ffs(active) - 1);
            const long long pos = (unsigned)base + rank;
            if (pos < out_cap) {
                out_build[pos] = build_row;
                out_probe[pos] = (int)i;
            }
        }
    }
}
