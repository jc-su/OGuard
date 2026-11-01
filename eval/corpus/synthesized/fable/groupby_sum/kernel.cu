// groupby_sum: hash aggregation of value by group into an open-addressing table.
//
// Home slot = mix64(group) & (capacity - 1) with linear probing. A row first
// reads the slot: a matching key gets an atomicAdd on the sum; an empty slot is
// claimed with atomicCAS (a lost race against the same key still lands on the
// right slot); any other key moves to the next slot. Keys never change once set,
// so the plain read before the CAS is safe.
//
// When capacity fits in shared memory (capacity <= SMEM_CAP) each block first
// aggregates into a block-private copy of the table, then flushes its non-empty
// slots into the global table with the same insert routine. This turns n global
// atomics into at most capacity per block. Larger tables use the global table
// directly.
//
// table_keys must be filled with -1 and table_sums with zero on entry.

#define SMEM_CAP 2048  // 2048 * (4 + 8) bytes = 24 KiB of shared memory per block

static __device__ __forceinline__ long long home_slot(int group, long long mask) {
    unsigned long long h = (unsigned long long)(long long)group;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return (long long)(h & (unsigned long long)mask);
}

static __device__ __forceinline__ void insert_add(int* keys, double* sums, int group, double v, long long mask) {
    long long slot = home_slot(group, mask);
    while (true) {
        const int cur = keys[slot];
        if (cur == group) {
            atomicAdd(&sums[slot], v);
            return;
        }
        if (cur == -1) {
            const int old = atomicCAS(&keys[slot], -1, group);
            if (old == -1 || old == group) {
                atomicAdd(&sums[slot], v);
                return;
            }
        }
        slot = (slot + 1) & mask;
    }
}

extern "C" __global__ void kernel(const int* __restrict__ group,
                                  const double* __restrict__ value,
                                  int* __restrict__ table_keys,
                                  double* __restrict__ table_sums,
                                  long long capacity,
                                  long long n) {
    const long long mask = capacity - 1;
    const long long stride = (long long)gridDim.x * blockDim.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (capacity <= SMEM_CAP) {
        __shared__ int local_keys[SMEM_CAP];
        __shared__ double local_sums[SMEM_CAP];
        for (long long j = threadIdx.x; j < capacity; j += blockDim.x) {
            local_keys[j] = -1;
            local_sums[j] = 0.0;
        }
        __syncthreads();
        for (; i < n; i += stride) insert_add(local_keys, local_sums, group[i], value[i], mask);
        __syncthreads();
        for (long long j = threadIdx.x; j < capacity; j += blockDim.x) {
            const int g = local_keys[j];
            if (g != -1) insert_add(table_keys, table_sums, g, local_sums[j], mask);
        }
    } else {
        for (; i < n; i += stride) insert_add(table_keys, table_sums, group[i], value[i], mask);
    }
}
