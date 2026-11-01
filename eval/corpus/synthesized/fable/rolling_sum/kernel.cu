// rolling_sum: trailing window sum of width `window` (1 to 1024).
//
// Each block produces 256 outputs. It loads its tile plus the window - 1
// preceding values (at most 1279 doubles) into shared memory, computes an
// inclusive prefix sum over them in up to five block-wide scan passes (warp
// shuffle scan plus a cross-warp carry), and forms every output as the
// difference of two prefix values, out[i] = P[i] - P[i - window]. That is
// two shared memory reads per output instead of `window` reads. The prefix
// sums are local to the block (at most 1279 terms), so the rounding error is
// of the same order as that of summing the window directly.

#define THREADS 256
#define MAX_WINDOW 1024
#define SMEM (THREADS + MAX_WINDOW)

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  double* __restrict__ out,
                                  long long n,
                                  long long window)
{
    __shared__ double s[SMEM];
    __shared__ double warp_sum[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const long long t0 = (long long)blockIdx.x * THREADS;
    if (t0 >= n) return;
    const int tile = (int)min((long long)THREADS, n - t0);
    const long long h = max(0LL, t0 - window + 1);  // first value the block needs
    const int e = (int)(t0 + tile - h);              // number of values to scan

    for (int l = tid; l < e; l += THREADS) s[l] = values[h + l];
    __syncthreads();

    // Inclusive prefix sum of s[0:e], 256 values per pass.
    double carry = 0.0;
    for (int base = 0; base < e; base += THREADS) {
        const int l = base + tid;
        double x = l < e ? s[l] : 0.0;
        for (int off = 1; off < 32; off <<= 1) {
            const double y = __shfl_up_sync(0xffffffffu, x, off);
            if (lane >= off) x += y;
        }
        if (lane == 31) warp_sum[wid] = x;
        __syncthreads();
        double prefix = carry;
        for (int w = 0; w < wid; ++w) prefix += warp_sum[w];
        double total = 0.0;
        for (int w = 0; w < 8; ++w) total += warp_sum[w];
        if (l < e) s[l] = x + prefix;
        carry += total;
        __syncthreads();
    }

    if (tid < tile) {
        const long long i = t0 + tid;
        const int l = (int)(i - h);
        double v = s[l];
        if (l >= window) v -= s[l - window];
        out[i] = v;
    }
}
