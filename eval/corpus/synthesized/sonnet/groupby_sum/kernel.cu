// GPU kernel for groupby_sum.
//
// Hash aggregation: sums value per group into an open-addressing
// table (table_keys, table_sums) with linear probing. capacity is a
// power of two, at least 2 * (number of distinct groups), so the
// table never fills.
//
// Precondition: the caller fills table_keys with -1 and zeroes
// table_sums before each launch.
//
// Concurrent insertion is race free: two threads racing to claim the
// same empty slot for the same group both see the winner's key (via
// atomicCAS or an equal already-installed key) and both add their
// value; threads with different, colliding groups simply retry at
// the next slot.

const int EMPTY = -1;

// Murmur3 finalizer: mixes a 32-bit key for table placement.
__device__ __forceinline__ unsigned int hash32(int key) {
    unsigned int h = (unsigned int)key;
    h ^= h >> 16;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return h;
}

extern "C" __global__ void kernel(const int *group, const double *value,
                                   int *table_keys, double *table_sums,
                                   long long capacity, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int g = group[i];
    double v = value[i];
    unsigned long long mask = (unsigned long long)(capacity - 1);
    long long slot = (long long)((unsigned long long)hash32(g) & mask);

    while (true) {
        int old = atomicCAS(&table_keys[slot], EMPTY, g);
        if (old == EMPTY || old == g) {
            atomicAdd(&table_sums[slot], v);
            break;
        }
        slot = (long long)(((unsigned long long)slot + 1) & mask);
    }
}
