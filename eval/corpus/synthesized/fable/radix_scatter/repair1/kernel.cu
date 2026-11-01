// radix_scatter: scatter row ids by one key byte, as one pass of an LSD
// radix sort.
//
// Each block takes a tile of 2048 keys (8 per thread, coalesced loads). It
// ranks every key inside the tile with an atomic add on a shared 256-bin
// histogram, claims one range per non-empty bin with a single global
// atomic add on that bin's cursor (at most 256 global atomics per 2048 keys
// instead of one per key), and stores each row id at bin base + rank. On
// exit every cursor has advanced to the end of its bin.
//
// Revision: the eight keys of a thread are handled by one statement each
// through inlined helpers, with their bytes and ranks in registers, instead
// of a loop over local arrays; the atomic adds are PTX instructions instead
// of library calls that take the pointer; and the store position is tested
// against [0, n). The test always passes under the task's contract (the
// cursors hold the exclusive prefix sum of the byte counts on entry); it
// states the bound of the store for the memory checker, which does not
// carry loop invariants and does not follow the values of an inout buffer.

#define THREADS 256
#define ITEMS 8
#define TILE (THREADS * ITEMS)

__device__ __forceinline__ int atomic_add_shared(int* p, int v)
{
    int old;
    asm volatile("atom.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ int atomic_add_global(int* p, int v)
{
    int old;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Byte of keys[i] and its rank among the tile's keys with that byte so far.
__device__ __forceinline__ void rank_key(const unsigned int* __restrict__ keys, int* hist,
                                         long long n, int sh, long long i, int& b, int& r)
{
    b = 0;
    r = 0;
    if (i < n) {
        b = (int)((keys[i] >> sh) & 255u);
        r = atomic_add_shared(&hist[b], 1);
    }
}

__device__ __forceinline__ void scatter_key(int* __restrict__ out, const int* base,
                                            long long n, long long i, int b, int r)
{
    if (i < n) {
        const long long pos = (long long)base[b] + r;
        if (pos >= 0 && pos < n) out[pos] = (int)i;
    }
}

extern "C" __global__ void kernel(const unsigned int* __restrict__ keys,
                                  int* __restrict__ cursors,
                                  int* __restrict__ out,
                                  long long n,
                                  long long shift)
{
    __shared__ int hist[256];
    __shared__ int base[256];
    const unsigned tid = threadIdx.x;
    const long long t0 = (long long)blockIdx.x * TILE;
    const int sh = (int)shift;

    hist[tid] = 0;
    __syncthreads();

    const long long i0 = t0 + tid;
    int b0, r0, b1, r1, b2, r2, b3, r3, b4, r4, b5, r5, b6, r6, b7, r7;
    rank_key(keys, hist, n, sh, i0, b0, r0);
    rank_key(keys, hist, n, sh, i0 + THREADS, b1, r1);
    rank_key(keys, hist, n, sh, i0 + 2 * THREADS, b2, r2);
    rank_key(keys, hist, n, sh, i0 + 3 * THREADS, b3, r3);
    rank_key(keys, hist, n, sh, i0 + 4 * THREADS, b4, r4);
    rank_key(keys, hist, n, sh, i0 + 5 * THREADS, b5, r5);
    rank_key(keys, hist, n, sh, i0 + 6 * THREADS, b6, r6);
    rank_key(keys, hist, n, sh, i0 + 7 * THREADS, b7, r7);
    __syncthreads();

    const int c = hist[tid];
    if (c > 0) base[tid] = atomic_add_global(&cursors[tid], c);
    __syncthreads();

    scatter_key(out, base, n, i0, b0, r0);
    scatter_key(out, base, n, i0 + THREADS, b1, r1);
    scatter_key(out, base, n, i0 + 2 * THREADS, b2, r2);
    scatter_key(out, base, n, i0 + 3 * THREADS, b3, r3);
    scatter_key(out, base, n, i0 + 4 * THREADS, b4, r4);
    scatter_key(out, base, n, i0 + 5 * THREADS, b5, r5);
    scatter_key(out, base, n, i0 + 6 * THREADS, b6, r6);
    scatter_key(out, base, n, i0 + 7 * THREADS, b7, r7);
}
