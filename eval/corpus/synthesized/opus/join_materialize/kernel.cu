// join_materialize: for probe row i whose key matches build rows f .. f + c - 1,
// out_build[offsets[i] + j] = f + j and out_probe[offsets[i] + j] = i.
//
// The work is split by output position, so a probe row with many matches does not
// serialize one thread. Each block takes tiles of 256 * ITEMS output positions; two
// threads find the first and last probe row of the tile by binary search over
// offsets, and each thread then finds the probe row of its first position within
// that range and writes ITEMS contiguous positions, moving to the next nonempty
// probe row when it reaches the end of one. f is the lower bound of probe[i] in
// the sorted build column, found once per probe row a thread touches.

#define ITEMS 8

// First index k in [lo, hi) with a[k] > x, or hi.
__device__ __forceinline__ long long upper_bound_i32(const int* __restrict__ a, long long lo,
                                                     long long hi, long long x)
{
    while (lo < hi) {
        const long long mid = lo + ((hi - lo) >> 1);
        if ((long long)a[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// First index k in [0, n) with a[k] >= x, or n.
__device__ __forceinline__ long long lower_bound_i64(const long long* __restrict__ a, long long n,
                                                     long long x)
{
    long long lo = 0, hi = n;
    while (lo < hi) {
        const long long mid = lo + ((hi - lo) >> 1);
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ void kernel(const long long* __restrict__ build,
                                  const long long* __restrict__ probe,
                                  const int* __restrict__ offsets,
                                  int* __restrict__ out_build,
                                  int* __restrict__ out_probe,
                                  long long b, long long p)
{
    __shared__ long long tile_rows[2];
    const long long total = offsets[p];
    const long long tile = (long long)blockDim.x * ITEMS;

    for (long long t0 = (long long)blockIdx.x * tile; t0 < total; t0 += (long long)gridDim.x * tile) {
        const long long t1 = t0 + tile < total ? t0 + tile : total;
        if (threadIdx.x < 2) {
            const long long target = threadIdx.x == 0 ? t0 : t1 - 1;
            tile_rows[threadIdx.x] = upper_bound_i32(offsets, 0, p + 1, target) - 1;
        }
        __syncthreads();
        const long long row_lo = tile_rows[0];
        const long long row_hi = tile_rows[1];
        __syncthreads();

        long long q = t0 + (long long)threadIdx.x * ITEMS;
        const long long q_end = q + ITEMS < t1 ? q + ITEMS : t1;
        if (q < q_end) {
            long long r = upper_bound_i32(offsets, row_lo, row_hi + 1, q) - 1;
            long long r_end = offsets[r + 1];
            long long f = lower_bound_i64(build, b, probe[r]) + (q - offsets[r]);
            for (; q < q_end; ++q) {
                if (q >= r_end) {
                    do {
                        ++r;
                        r_end = offsets[r + 1];
                    } while (q >= r_end);
                    f = lower_bound_i64(build, b, probe[r]);
                }
                out_build[q] = (int)f++;
                out_probe[q] = (int)r;
            }
        }
    }
}
