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

#define THREADS 256
#define ITEMS 4
#define TILE (THREADS * ITEMS)

// Number of elements of a[0:na] among the first `diag` merged outputs.
__device__ __forceinline__ long long merge_path(const long long* a, long long na,
                                                const long long* b, long long nb,
                                                long long diag)
{
    long long lo = diag > nb ? diag - nb : 0;
    long long hi = diag < na ? diag : na;
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (a[mid] <= b[diag - 1 - mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ void kernel(const long long* __restrict__ a,
                                  const long long* __restrict__ b,
                                  long long* __restrict__ out,
                                  long long m,
                                  long long p)
{
    __shared__ long long s[TILE];
    __shared__ long long merged[TILE];
    const int tid = threadIdx.x;
    const long long total = m + p;
    const long long k0 = (long long)blockIdx.x * TILE;
    if (k0 >= total) return;
    const long long k1 = min(k0 + TILE, total);
    const int tile = (int)(k1 - k0);

    const long long a0 = merge_path(a, m, b, p, k0);
    const long long a1 = merge_path(a, m, b, p, k1);
    const long long b0 = k0 - a0;
    const int na = (int)(a1 - a0);
    const int nb = (int)(k1 - a1 - b0);

    for (int l = tid; l < tile; l += THREADS) s[l] = l < na ? a[a0 + l] : b[b0 + l - na];
    __syncthreads();

    const int d = tid * ITEMS;
    if (d < tile) {
        const int d1 = min(d + ITEMS, tile);
        int i = (int)merge_path(s, na, s + na, nb, d);
        int j = d - i;
        for (int k = d; k < d1; ++k) {
            if (j >= nb || (i < na && s[i] <= s[na + j])) merged[k] = s[i++];
            else merged[k] = s[na + j++];
        }
    }
    __syncthreads();

    for (int l = tid; l < tile; l += THREADS) out[k0 + l] = merged[l];
}
