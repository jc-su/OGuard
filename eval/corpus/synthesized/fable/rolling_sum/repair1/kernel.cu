// rolling_sum: trailing window sum of width `window` (1 to 1024).
//
// Each block produces 256 outputs. It stages its tile plus the window - 1
// preceding values (at most 1279 doubles) in shared memory, computes an
// inclusive prefix sum over them, and forms every output as the difference
// of two prefix values, out[i] = P[i] - P[i - window]. That is two shared
// memory reads per output instead of `window` reads. The prefix sums are
// local to the block (at most 1279 terms), so the rounding error is of the
// same order as that of summing the window directly.
//
// Revision: the prefix sum is one block scan of five consecutive values per
// thread instead of five strided scan passes, so no loop indexes memory: the
// staging, the scan and the write-back are one statement per value, and the
// cross-warp step is a shuffle scan. The staged length is capped at the
// buffer size (a no-op, as window <= 1024) and the output index is tested
// against the staged range. The tests always pass under the task's
// contract; they state the bound of each access for the memory checker,
// which does not carry loop invariants.

#define THREADS 256
#define MAX_WINDOW 1024
#define SMEM (THREADS + MAX_WINDOW)  // the tile plus the halo of the widest window
#define ITEMS (SMEM / THREADS)       // consecutive staged values per thread in the scan

__device__ __forceinline__ void stage(double* s, const double* __restrict__ values,
                                      long long h, long long e, long long l)
{
    if (l < e) s[l] = values[h + l];
}

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  double* __restrict__ out,
                                  long long n,
                                  long long window)
{
    __shared__ double s[SMEM];
    __shared__ double warp_sum[8];
    __shared__ double warp_offset[8];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31;
    const unsigned wid = tid >> 5;
    const long long t0 = (long long)blockIdx.x * THREADS;
    if (t0 >= n) return;
    const long long tile = n - t0 < THREADS ? n - t0 : THREADS;
    const long long h = t0 - window + 1 > 0 ? t0 - window + 1 : 0;  // first value the block needs
    const long long e0 = t0 + tile - h;
    const long long e = e0 < SMEM ? e0 : SMEM;  // number of values to stage

    // s[0:e] = values[h:h + e], coalesced.
    stage(s, values, h, e, tid);
    stage(s, values, h, e, tid + THREADS);
    stage(s, values, h, e, tid + 2 * THREADS);
    stage(s, values, h, e, tid + 3 * THREADS);
    stage(s, values, h, e, tid + 4 * THREADS);
    __syncthreads();

    // Inclusive prefix sum of s[0:e]: each thread scans ITEMS consecutive
    // values in registers, the block scans the thread totals.
    const long long b = (long long)tid * ITEMS;
    const double x0 = b < e ? s[b] : 0.0;
    const double x1 = b + 1 < e ? s[b + 1] : 0.0;
    const double x2 = b + 2 < e ? s[b + 2] : 0.0;
    const double x3 = b + 3 < e ? s[b + 3] : 0.0;
    const double x4 = b + 4 < e ? s[b + 4] : 0.0;
    const double p0 = x0;
    const double p1 = p0 + x1;
    const double p2 = p1 + x2;
    const double p3 = p2 + x3;
    const double p4 = p3 + x4;

    double x = p4;
    for (int off = 1; off < 32; off <<= 1) {
        const double y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warp_sum[wid] = x;
    __syncthreads();
    if (tid < 32) {
        const double c = tid < 8 ? warp_sum[tid] : 0.0;
        double z = c;
        for (int off = 1; off < 32; off <<= 1) {
            const double y = __shfl_up_sync(0xffffffffu, z, off);
            if (lane >= off) z += y;
        }
        if (tid < 8) warp_offset[tid] = z - c;
    }
    __syncthreads();

    const double prefix = warp_offset[wid] + x - p4;  // sum of s[0:b]
    if (b < e) s[b] = prefix + p0;
    if (b + 1 < e) s[b + 1] = prefix + p1;
    if (b + 2 < e) s[b + 2] = prefix + p2;
    if (b + 3 < e) s[b + 3] = prefix + p3;
    if (b + 4 < e) s[b + 4] = prefix + p4;
    __syncthreads();

    const long long i = t0 + tid;
    if (i < n) {
        const long long l = i - h;
        if (l >= 0 && l < e) {
            double v = s[l];
            const long long lw = l - window;
            if (lw >= 0 && lw < e) v -= s[lw];
            out[i] = v;
        }
    }
}
