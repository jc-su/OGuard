// group_argmax: best[k] = smallest row of group k whose value is the largest of the
// group; stays -1 for a group without rows.
//
// best[k] holds a row; candidates are ordered by (value desc, row asc) and the row is
// replaced by compare-and-swap, comparing against value[best[k]], so best[k] only
// improves. A thread first reads best[k] and skips the CAS when its candidate does
// not win; a stale read only shows an older (worse) row, so the skip is safe.
// To cut contention when many rows share a group (few groups, or sorted groups),
// each warp groups its lanes by group id with __match_any_sync. When the warp holds
// at most MAX_WARP_GROUPS distinct groups, each group is reduced with shuffles and
// its leader issues one update; otherwise every lane updates on its own.

#define MAX_WARP_GROUPS 4

__device__ __forceinline__ bool wins(long long v, long long r, long long bv, long long br)
{
    return v > bv || (v == bv && r < br);
}

__device__ __forceinline__ void update(int* best, const long long* __restrict__ value, int k, long long v,
                                       int r)
{
    int cur = best[k];
    while (cur < 0 || wins(v, r, value[cur], cur)) {
        const int old = atomicCAS(&best[k], cur, r);
        if (old == cur) return;
        cur = old;
    }
}

extern "C" __global__ void kernel(const int* __restrict__ group,
                                  const long long* __restrict__ value,
                                  int* best,
                                  long long n, long long g)
{
    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const long long stride = (long long)gridDim.x * blockDim.x;
    // base is the same for all lanes of a warp, so the loop is warp-uniform.
    for (long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) - lane; base < n;
         base += stride) {
        const long long i = base + lane;
        const bool in_range = i < n;
        const int k = in_range ? group[i] : -1;
        const long long v = in_range ? value[i] : 0;
        const unsigned int peers = __match_any_sync(full, k);
        const bool leader = lane == __ffs(peers) - 1;
        unsigned int leaders = __ballot_sync(full, leader);
        if (__popc(leaders) <= MAX_WARP_GROUPS) {
            while (leaders) {
                const int src = __ffs(leaders) - 1;
                leaders &= leaders - 1;
                const int gk = __shfl_sync(full, k, src);
                const bool mine = k == gk;
                long long bv = mine ? v : (-0x7fffffffffffffffll - 1);
                long long br = mine ? i : 0x7fffffffffffffffll;
                for (int d = 16; d > 0; d >>= 1) {
                    const long long ov = __shfl_xor_sync(full, bv, d);
                    const long long orow = __shfl_xor_sync(full, br, d);
                    if (wins(ov, orow, bv, br)) {
                        bv = ov;
                        br = orow;
                    }
                }
                if (lane == src && gk >= 0) update(best, value, gk, bv, (int)br);
            }
        } else if (in_range) {
            update(best, value, k, v, (int)i);
        }
    }
}
