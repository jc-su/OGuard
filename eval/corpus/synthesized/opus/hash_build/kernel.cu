// Hash build: insert each key into an open-addressing table with linear probing.
//
// Slot s holds a key in table_keys[s] and the key's row id in table_rows[s].
// The home slot of a key is fmix64(key) & (capacity - 1). A key goes into the
// first empty slot at or after its home slot, wrapping around the table end.
// hash_probe uses the same hash and probe order.
//
// Preconditions: keys are distinct and not -1, capacity is a power of two and
// at least max(1, 2 * n), and every table_keys slot holds -1 on entry.

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

extern "C" __global__ void kernel(const long long* __restrict__ keys,
                                  long long* __restrict__ table_keys,
                                  int* __restrict__ table_rows,
                                  long long capacity,
                                  long long n)
{
    const unsigned long long mask = static_cast<unsigned long long>(capacity - 1);
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const long long key = keys[i];
        unsigned long long slot = hash_key(key) & mask;
        while (true) {
            // A slot changes only from kEmpty to a key, so a stale read of a key
            // is never wrong; the CAS settles a stale read of kEmpty.
            if (table_keys[slot] == kEmpty) {
                const unsigned long long previous =
                    atomicCAS(reinterpret_cast<unsigned long long*>(table_keys + slot),
                              static_cast<unsigned long long>(kEmpty),
                              static_cast<unsigned long long>(key));
                if (static_cast<long long>(previous) == kEmpty) {
                    table_rows[slot] = static_cast<int>(i);
                    break;
                }
            }
            slot = (slot + 1) & mask;
        }
    }
}
