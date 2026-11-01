// GPU kernel for hash_build.
//
// Inserts each row of keys into an open-addressing hash table with
// linear probing. capacity is a power of two, at least 2 * n, so the
// table never fills and every probe sequence is guaranteed to reach
// an empty slot.
//
// Precondition: the caller fills table_keys with -1 before each
// launch.

const long long EMPTY = -1;

// splitmix64 finalizer: mixes a 64-bit key for table placement.
__device__ __forceinline__ unsigned long long hash64(long long key) {
    unsigned long long h = (unsigned long long)key;
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h = h ^ (h >> 31);
    return h;
}

extern "C" __global__ void kernel(const long long *keys, long long *table_keys,
                                   int *table_rows, long long capacity,
                                   long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    long long key = keys[i];
    unsigned long long mask = (unsigned long long)(capacity - 1);
    long long slot = (long long)(hash64(key) & mask);

    while (true) {
        long long old =
            (long long)atomicCAS((unsigned long long *)&table_keys[slot],
                                  (unsigned long long)EMPTY,
                                  (unsigned long long)key);
        if (old == EMPTY) {
            table_rows[slot] = (int)i;
            break;
        }
        slot = (long long)(((unsigned long long)slot + 1) & mask);
    }
}
