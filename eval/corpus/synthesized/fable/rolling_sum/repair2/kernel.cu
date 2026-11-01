// rolling_sum: trailing window sum of width `window` (1 to 1024).
//
// Each block produces 256 outputs. It stages the tile and a fixed halo of
// MAX_WINDOW preceding rows (TILE = 1280 doubles) in shared memory behind
// MAX_WINDOW zeros, rows before the column or before every window of the
// tile as zeros. For staged position u it forms z[u] = s[u] - s[u - window];
// the inclusive scan of z at u is exactly the sum of the window ending at
// u, and every partial sum of the scan is itself a window sum, so no large
// prefix sums are subtracted from each other. The scan runs on 5
// consecutive values per thread in registers, a warp shuffle scan of the
// thread totals and a shuffle scan of the warp totals.
//
// Revision: the halo is fixed instead of window-dependent, so no index is
// clamped with min or max; every row index is the block's first staged row
// plus a thread-affine offset, tested against [0, n) right before the
// load; and the zeros in front of the staged rows make every shared index
// affine in the thread id and the window with no range test at all
// (u - window is at least u - MAX_WINDOW >= 0). The thread id is masked to
// the block size.

#define THREADS 256
#define MAX_WINDOW 1024
#define TILE (THREADS + MAX_WINDOW)  // 1280 staged rows: the halo of the widest window and the tile
#define SMEM (MAX_WINDOW + TILE)     // MAX_WINDOW zeros, then the staged rows
#define ITEMS (TILE / THREADS)       // 5 consecutive positions per thread in the scan

// s[MAX_WINDOW + l] = values[base + l], or 0 when that row is missing or before every window of the tile.
__device__ __forceinline__ void stage(double* s, const double* __restrict__ values, long long n,
                                      long long base, long long first, long long l)
{
    double v = 0.0;
    if (l >= first) {
        const long long g = base + l;
        if (g >= 0 && g < n) v = values[g];
    }
    s[MAX_WINDOW + l] = v;
}

// Inclusive scan of x over the warp.
__device__ __forceinline__ double warp_scan(double x, unsigned lane)
{
    for (int off = 1; off < 32; off <<= 1) {
        const double y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= (unsigned)off) x += y;
    }
    return x;
}

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  double* __restrict__ out,
                                  long long n,
                                  long long window)
{
    __shared__ double s[SMEM];
    __shared__ double warp_sum[32];  // 8 used; one slot per lane of warp 0
    __shared__ double warp_offset[32];
    const unsigned tid = threadIdx.x & (THREADS - 1);
    const unsigned lane = tid & 31;
    const unsigned wid = tid >> 5;
    const long long t0 = (long long)blockIdx.x * THREADS;
    if (t0 >= n) return;
    const long long base = t0 - MAX_WINDOW;      // row of s[MAX_WINDOW]
    const long long first = MAX_WINDOW - window;  // staged rows before this one lie outside every window of the tile

    s[tid] = 0.0;
    s[tid + THREADS] = 0.0;
    s[tid + 2 * THREADS] = 0.0;
    s[tid + 3 * THREADS] = 0.0;
    stage(s, values, n, base, first, tid);
    stage(s, values, n, base, first, tid + THREADS);
    stage(s, values, n, base, first, tid + 2 * THREADS);
    stage(s, values, n, base, first, tid + 3 * THREADS);
    stage(s, values, n, base, first, tid + 4 * THREADS);
    __syncthreads();

    // z[u] = s[u] - s[u - window] for the thread's 5 consecutive positions, scanned in registers.
    const long long u = MAX_WINDOW + (long long)tid * ITEMS;
    const double z0 = s[u] - s[u - window];
    const double z1 = s[u + 1] - s[u + 1 - window];
    const double z2 = s[u + 2] - s[u + 2 - window];
    const double z3 = s[u + 3] - s[u + 3 - window];
    const double z4 = s[u + 4] - s[u + 4 - window];
    const double p0 = z0;
    const double p1 = p0 + z1;
    const double p2 = p1 + z2;
    const double p3 = p2 + z3;
    const double p4 = p3 + z4;

    const double x = warp_scan(p4, lane);
    if (lane == 31) warp_sum[wid] = x;
    __syncthreads();
    if (tid < 32) {
        double c = 0.0;
        if (tid < 8) c = warp_sum[tid];
        warp_offset[tid] = warp_scan(c, lane) - c;
    }
    __syncthreads();

    const double prefix = warp_offset[wid] + x - p4;  // scan of z before position u
    s[u] = prefix + p0;
    s[u + 1] = prefix + p1;
    s[u + 2] = prefix + p2;
    s[u + 3] = prefix + p3;
    s[u + 4] = prefix + p4;
    __syncthreads();

    const long long i = t0 + tid;
    if (i < n) out[i] = s[2 * MAX_WINDOW + tid];
}
