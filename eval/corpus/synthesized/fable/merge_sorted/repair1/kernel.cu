// merge_sorted: merge two sorted int64 columns, a before an equal b.
//
// Merge path (Green, McColl, Bader / ModernGPU). Each block owns 1024
// consecutive outputs. Every thread finds the block's slice of `a` and `b`
// with two merge-path searches on the global arrays (a binary search along
// the diagonal; ties are resolved so that a precedes an equal b), the block
// loads the two slices into shared memory with coalesced loads, each thread
// finds its own 4-output slice with a merge-path search in shared memory and
// merges it sequentially into a shared staging buffer, and the block writes
// the staging buffer to `out` with coalesced stores.
//
// Revision: every probe of the searches and every element read of the merge
// is guarded by an explicit range test on its index against the size of the
// array it reads, the loops over the block's slice are one statement per
// element, and the output index is tested against m + p. The tests always
// pass under the task's contract (both inputs are sorted, so the merge path
// stays inside them); they state the bound of each access for the memory
// checker, which does not follow calls or carry loop invariants.

#define THREADS 256
#define ITEMS 4
#define TILE (THREADS * ITEMS)

// Number of elements of a[0:na] among the first `diag` outputs of merging
// a[0:na] with b[b_off:b_off + nb]; a_len and b_len are the sizes of a and b.
__device__ __forceinline__ long long merge_path(const long long* a, long long a_len,
                                                const long long* b, long long b_off, long long b_len,
                                                long long na, long long nb, long long diag)
{
    long long lo = diag > nb ? diag - nb : 0;
    long long hi = diag < na ? diag : na;
    while (lo < hi) {
        // the unsigned shift keeps mid >= 0
        const long long mid = (long long)(((unsigned long long)lo + (unsigned long long)hi) >> 1);
        const long long jb = b_off + diag - 1 - mid;
        if (mid >= a_len || jb < 0 || jb >= b_len) break;
        if (a[mid] <= b[jb]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// s[l] = element l of the block's slices, a[a0:a0 + na] then b[b0:b0 + tile - na].
__device__ __forceinline__ void load_slice(long long* s, const long long* __restrict__ a,
                                           const long long* __restrict__ b, long long m, long long p,
                                           long long a0, long long b0, long long na, long long tile,
                                           long long l)
{
    if (l < tile) {
        if (l < na) {
            const long long ia = a0 + l;
            if (ia >= 0 && ia < m) s[l] = a[ia];
        } else {
            const long long ib = b0 + l - na;
            if (ib >= 0 && ib < p) s[l] = b[ib];
        }
    }
}

// Output k of the thread's slice: s[i] (from a) or s[na + j] (from b), a first on ties.
__device__ __forceinline__ void merge_step(const long long* s, long long* merged, long long na,
                                           long long nb, long long tile, long long& i, long long& j,
                                           long long k)
{
    if (k < tile) {
        const long long ib = na + j;
        const bool has_a = i < na && i >= 0 && i < TILE;
        const bool has_b = j < nb && ib >= 0 && ib < TILE;
        if (has_a && (!has_b || s[i] <= s[ib])) {
            merged[k] = s[i];
            ++i;
        } else if (has_b) {
            merged[k] = s[ib];
            ++j;
        }
    }
}

__device__ __forceinline__ void store_slice(long long* __restrict__ out, const long long* merged,
                                            long long k0, long long total, long long l)
{
    const long long g = k0 + l;
    if (g < total) out[g] = merged[l];
}

extern "C" __global__ void kernel(const long long* __restrict__ a,
                                  const long long* __restrict__ b,
                                  long long* __restrict__ out,
                                  long long m,
                                  long long p)
{
    __shared__ long long s[TILE];
    __shared__ long long merged[TILE];
    const unsigned tid = threadIdx.x;
    const long long total = m + p;
    const long long k0 = (long long)blockIdx.x * TILE;
    if (k0 >= total) return;
    const long long k1 = k0 + TILE < total ? k0 + TILE : total;
    const long long tile = k1 - k0;

    const long long a0 = merge_path(a, m, b, 0, p, m, p, k0);
    const long long a1 = merge_path(a, m, b, 0, p, m, p, k1);
    const long long b0 = k0 - a0;
    const long long na = a1 - a0;
    const long long nb = tile - na;

    load_slice(s, a, b, m, p, a0, b0, na, tile, tid);
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + THREADS);
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + 2 * THREADS);
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + 3 * THREADS);
    __syncthreads();

    const long long d = (long long)tid * ITEMS;
    if (d < tile) {
        long long i = merge_path(s, TILE, s, na, TILE, na, nb, d);
        long long j = d - i;
        merge_step(s, merged, na, nb, tile, i, j, d);
        merge_step(s, merged, na, nb, tile, i, j, d + 1);
        merge_step(s, merged, na, nb, tile, i, j, d + 2);
        merge_step(s, merged, na, nb, tile, i, j, d + 3);
    }
    __syncthreads();

    store_slice(out, merged, k0, total, tid);
    store_slice(out, merged, k0, total, tid + THREADS);
    store_slice(out, merged, k0, total, tid + 2 * THREADS);
    store_slice(out, merged, k0, total, tid + 3 * THREADS);
}
