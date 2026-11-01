// argmin: row[0] = smallest i such that values[i] is the minimum of values; stays -1
// for n = 0.
//
// Each block reduces its rows to the best (value, row) pair, ordered by value and
// then by row, and publishes it with a compare-and-swap loop on row[0]. The loop
// compares against values[row[0]], so the published row only ever improves; a
// stale read of row[0] only causes a failed CAS and a retry with the fresh value.

__device__ __forceinline__ bool better(long long v, long long r, long long bv, long long br)
{
    return v < bv || (v == bv && r < br);
}

extern "C" __global__ void kernel(const long long* __restrict__ values, int* row, long long n)
{
    __shared__ long long s_val[32];
    __shared__ long long s_row[32];
    const unsigned int full = 0xffffffffu;
    const long long none = 0x7fffffffffffffffll;  // row sentinel: loses every tie

    long long best_v = 0x7fffffffffffffffll;
    long long best_r = none;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long v = values[i];
        if (better(v, i, best_v, best_r)) {
            best_v = v;
            best_r = i;
        }
    }
    for (int d = 16; d > 0; d >>= 1) {
        const long long ov = __shfl_xor_sync(full, best_v, d);
        const long long orow = __shfl_xor_sync(full, best_r, d);
        if (better(ov, orow, best_v, best_r)) {
            best_v = ov;
            best_r = orow;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) {
        s_val[warp] = best_v;
        s_row[warp] = best_r;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int w = 1; w < warps; ++w) {
        if (better(s_val[w], s_row[w], best_v, best_r)) {
            best_v = s_val[w];
            best_r = s_row[w];
        }
    }
    if (best_r == none) return;  // block had no rows

    int cur = *(volatile int*)row;
    while (true) {
        if (cur >= 0 && !better(best_v, best_r, values[cur], cur)) return;
        const int old = atomicCAS(row, cur, (int)best_r);
        if (old == cur) return;
        cur = old;
    }
}
