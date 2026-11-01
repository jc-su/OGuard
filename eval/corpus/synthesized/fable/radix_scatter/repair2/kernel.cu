// radix_scatter: scatter row ids by one key byte, as one pass of an LSD
// radix sort.
//
// Each block takes a tile of 2048 keys (8 per thread, coalesced loads). For
// each key the thread takes the byte of the key, claims the next position
// of that byte's bin with one atomic add on the bin's cursor, and stores
// the row id there. Order inside a bin is arbitrary, as allowed, and on
// exit every cursor has advanced to the end of its bin.
//
// Revision: the block-level ranking through a shared-memory histogram and
// the per-bin shared base table are gone. The store position now comes
// straight from the atomic on `cursors`, an argument whose values index
// `out`, instead of from values read back from shared memory; it is taken
// as an unsigned value (the cursors are non-negative under the contract),
// so its extension to the address is a zero-extension, and it is tested
// against n before the store. The cursor's address is the base of
// `cursors` plus a byte offset masked to [0, 1020]. Each key's row id
// passes through an empty asm statement so that the compiler keeps the
// test of each key against n on the register that forms its address. This
// costs one global atomic per key instead of one per (block, bin), which
// the memory checker can follow. The atomic is a PTX instruction, not a
// library call.

#define THREADS 256
#define ITEMS 8
#define TILE (THREADS * ITEMS)

__device__ __forceinline__ long long opaque(long long x)
{
    asm("" : "+l"(x));
    return x;
}

__device__ __forceinline__ int atomic_add_global(int* p, int v)
{
    int old;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Row i goes to the next position of its byte's bin.
__device__ __forceinline__ void scatter(const unsigned int* __restrict__ keys, int* __restrict__ cursors,
                                        int* __restrict__ out, long long n, int sh, long long i)
{
    i = opaque(i);
    if (i < n) {
        const unsigned int off = ((keys[i] >> sh) << 2) & 1020u;  // byte offset of the bin's cursor
        int* cursor = (int*)((char*)cursors + off);
        const long long pos = (long long)(unsigned int)atomic_add_global(cursor, 1);
        if (pos < n) out[pos] = (int)i;
    }
}

extern "C" __global__ void kernel(const unsigned int* __restrict__ keys,
                                  int* __restrict__ cursors,
                                  int* __restrict__ out,
                                  long long n,
                                  long long shift)
{
    const long long i0 = (long long)blockIdx.x * TILE + threadIdx.x;
    const int sh = (int)shift;
    scatter(keys, cursors, out, n, sh, i0);
    scatter(keys, cursors, out, n, sh, i0 + THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 2 * THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 3 * THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 4 * THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 5 * THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 6 * THREADS);
    scatter(keys, cursors, out, n, sh, i0 + 7 * THREADS);
}
