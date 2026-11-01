// GPU kernel for hash_probe.
//
// Probes the open-addressing hash table built by hash_build
// (table_keys, table_rows, capacity) with probe_keys, and for each
// match emits the pair (build row id, probe row id) into out_build /
// out_probe.
//
// Precondition: the caller zeroes out_count before each launch.
//
// Build keys are distinct, so at most one build row matches each
// probe key: the probe stops at the first match. An empty slot
// (table_keys == -1) is reached within at most capacity steps,
// because capacity is at least 2 * (number of build keys), so the
// table is never full.
//
// out_count is an output buffer, so its value on entry is not part of
// the task's contract: every store must be safe for whatever
// out_count already holds, not only for the zeroed value the caller
// is asked to supply. The prior guard, "pos < out_cap", only bounds
// pos from above; if out_count started negative the atomic add could
// still return a negative pos that passes that check and stores
// below the start of out_build/out_probe. The guard now checks both
// ends of the range before the store.

const long long EMPTY = -1;

// splitmix64 finalizer: mixes a 64-bit key for table placement.
__device__ __forceinline__ unsigned long long hash64(long long key) {
    unsigned long long h = (unsigned long long)key;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h = h ^ (h >> 31);
    return h;
}

extern "C" __global__ void kernel(const long long *probe_keys,
                                   const long long *table_keys,
                                   const int *table_rows, int *out_build,
                                   int *out_probe, int *out_count,
                                   long long capacity, long long m,
                                   long long out_cap) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;

    long long key = probe_keys[j];
    unsigned long long mask = (unsigned long long)(capacity - 1);
    long long slot = (long long)(hash64(key) & mask);

    while (table_keys[slot] != EMPTY) {
        if (table_keys[slot] == key) {
            int pos = atomicAdd(out_count, 1);
            if (pos >= 0 && pos < out_cap) {
                out_build[pos] = table_rows[slot];
                out_probe[pos] = (int)j;
            }
            break;
        }
        slot = (long long)(((unsigned long long)slot + 1) & mask);
    }
}
