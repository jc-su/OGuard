// merge_sorted: merge two sorted int64 columns, a before an equal b.
//
// Merge path (Green, McColl, Bader / ModernGPU) straight from global memory.
// Each thread owns ITEMS consecutive outputs that start at diagonal d. It
// finds the partition of that diagonal, the number of elements of `a` among
// the first d outputs, with a binary search over the diagonal, then merges
// its ITEMS outputs sequentially from `a` and `b`. Neighbouring threads
// probe neighbouring positions, so the search loads coalesce and the upper
// levels of the search stay in L1.
//
// Revision: the staging of tiles in shared memory and the block-level
// searches are gone, so no index that reaches memory passes through shared
// memory. The search runs over the whole diagonal and treats a probe past
// the end of an input as an exhausted input (a[m..] and b[p..] read as
// +inf), which removes the min/max clamps of the search bounds; it is
// indexed by k = i + 1 so that both probe indices, k - 1 and d - k, are
// plain sums. Every probe and every merge step tests the exact index it is
// about to use against 0 and the size of its array, and each element is
// loaded once; under the task's contract those tests always pass. The
// indices pass through an empty asm statement so that the compiler keeps
// those tests on the very register that forms the address.

#define THREADS 256
#define ITEMS 4

__device__ __forceinline__ long long opaque(long long x)
{
    asm("" : "+l"(x));
    return x;
}

// Output k of the merge: a[i] or b[j], a first on ties; advances i or j.
__device__ __forceinline__ void merge_step(const long long* __restrict__ a,
                                           const long long* __restrict__ b,
                                           long long* __restrict__ out,
                                           long long m, long long p, long long total,
                                           long long& i, long long& j, long long k)
{
    if (k < total) {
        const bool has_a = i >= 0 && i < m;
        const bool has_b = j >= 0 && j < p;
        long long av = 0, bv = 0;
        if (has_a) av = a[i];
        if (has_b) bv = b[j];
        if (has_a && (!has_b || av <= bv)) {
            out[k] = av;
            i = opaque(i + 1);
        } else if (has_b) {
            out[k] = bv;
            j = opaque(j + 1);
        }
    }
}

extern "C" __global__ void kernel(const long long* __restrict__ a,
                                  const long long* __restrict__ b,
                                  long long* __restrict__ out,
                                  long long m,
                                  long long p)
{
    const long long total = m + p;
    const long long d = ((long long)blockIdx.x * THREADS + threadIdx.x) * ITEMS;
    if (d >= total) return;

    // Smallest k in [1, d + 1] with a[k - 1] > b[d - k]; k - 1 is the a-count of the first d outputs.
    long long lo = 1, hi = d + 1;
    while (lo < hi) {
        const long long mid = opaque((lo + hi) >> 1);
        const long long ia = opaque(mid - 1);
        const long long jb = opaque(d - mid);
        if (ia >= m) hi = mid;             // a is exhausted at ia: the partition lies to the left
        else if (jb >= p) lo = mid + 1;    // b is exhausted at jb: the partition lies to the right
        else if (ia < 0) break;            // never under the contract
        else if (jb < 0) hi = mid;         // never under the contract
        else if (a[ia] <= b[jb]) lo = mid + 1;
        else hi = mid;
    }

    long long i = opaque(lo - 1);
    long long j = opaque(d - i);
    merge_step(a, b, out, m, p, total, i, j, d);
    merge_step(a, b, out, m, p, total, i, j, d + 1);
    merge_step(a, b, out, m, p, total, i, j, d + 2);
    merge_step(a, b, out, m, p, total, i, j, d + 3);
}
