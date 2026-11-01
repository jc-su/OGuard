// hash_build: insert (key, row id) into an open-addressing table with linear probing.
//
// Home slot = mix64(key) & (capacity - 1), where mix64 is the MurmurHash3 64-bit
// finalizer, so nearby keys spread over the table. A thread reads a slot first
// and issues the atomicCAS only on an empty slot; when the CAS succeeds the thread
// owns the slot and writes the row id. Duplicate keys, if any, get separate
// slots, which is the multimap semantics a join build needs.
//
// table_keys must be filled with -1 on entry.

static __device__ __forceinline__ unsigned long long mix64(long long key) {
    unsigned long long h = (unsigned long long)key;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(const long long* __restrict__ keys,
                                  long long* __restrict__ table_keys,
                                  int* __restrict__ table_rows,
                                  long long capacity,
                                  long long n) {
    const unsigned long long mask = (unsigned long long)(capacity - 1);
    const unsigned long long empty = (unsigned long long)-1LL;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long key = keys[i];
        unsigned long long slot = mix64(key) & mask;
        while (true) {
            if (table_keys[slot] == -1LL) {
                unsigned long long* p = reinterpret_cast<unsigned long long*>(table_keys + slot);
                if (atomicCAS(p, empty, (unsigned long long)key) == empty) {
                    table_rows[slot] = (int)i;
                    break;
                }
            }
            slot = (slot + 1) & mask;
        }
    }
}
