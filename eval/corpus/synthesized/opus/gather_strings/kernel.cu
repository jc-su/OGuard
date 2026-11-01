// gather_strings: out_chars[out_offsets[i] : out_offsets[i + 1]] = bytes of string idx[i].
//
// The work is split by output byte, so a long string does not serialize one
// thread. Each block takes tiles of 256 * BYTES_PER_THREAD output bytes; two
// threads find the first and last output row of the tile by binary search over
// out_offsets, and each thread then finds the row of its first byte within that
// range and copies BYTES_PER_THREAD contiguous bytes, moving to the next row
// (skipping empty rows) when it reaches the end of one.

#define BYTES_PER_THREAD 16

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

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const unsigned char* __restrict__ chars,
                                  const int* __restrict__ idx,
                                  const int* __restrict__ out_offsets,
                                  unsigned char* __restrict__ out_chars,
                                  long long n, long long m)
{
    __shared__ long long tile_rows[2];
    const long long total = out_offsets[m];
    const long long tile = (long long)blockDim.x * BYTES_PER_THREAD;

    for (long long t0 = (long long)blockIdx.x * tile; t0 < total; t0 += (long long)gridDim.x * tile) {
        const long long t1 = t0 + tile < total ? t0 + tile : total;
        if (threadIdx.x < 2) {
            const long long target = threadIdx.x == 0 ? t0 : t1 - 1;
            tile_rows[threadIdx.x] = upper_bound_i32(out_offsets, 0, m + 1, target) - 1;
        }
        __syncthreads();
        const long long row_lo = tile_rows[0];
        const long long row_hi = tile_rows[1];
        __syncthreads();

        long long p = t0 + (long long)threadIdx.x * BYTES_PER_THREAD;
        const long long p_end = p + BYTES_PER_THREAD < t1 ? p + BYTES_PER_THREAD : t1;
        if (p < p_end) {
            long long r = upper_bound_i32(out_offsets, row_lo, row_hi + 1, p) - 1;
            long long r_end = out_offsets[r + 1];
            long long src = (long long)offsets[idx[r]] + (p - out_offsets[r]);
            for (; p < p_end; ++p) {
                while (p >= r_end) {
                    ++r;
                    r_end = out_offsets[r + 1];
                    src = offsets[idx[r]];
                }
                out_chars[p] = chars[src++];
            }
        }
    }
}
