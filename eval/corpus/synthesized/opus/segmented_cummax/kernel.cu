// segmented_cummax: out[i] = max(values[first[i] .. i]), a running maximum that restarts
// at every partition start (first[i] == i).
//
// A partition can span the whole column, so the kernel is a cooperative launch (all
// blocks resident, grid.sync between phases) and block b owns rows
// [begin, end) = [b * chunk, (b + 1) * chunk):
//   1. Each block computes the maximum of the partition part that ends its rows,
//      values[max(first[end - 1], begin) .. end), and stores it in out[end - 1]
//      (overwritten in phase 3).
//   2. If its first row continues a partition (first[begin] < begin), a block
//      looks back over the previous blocks, 256 at a time, and takes the maximum
//      of their stored values up to the nearest block in which that partition
//      starts (first[end_k - 1] >= begin_k). That is the carry into its rows.
//   3. Each block scans its rows in tiles of 256 with a segmented max scan (warp
//      shuffles, then warp totals in shared memory), combining rows before the
//      first partition start of the tile with the carry, and writes out.

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MAX_WARPS 32
#define NEG_INF (-0x7fffffffffffffffll - 1)

__device__ __forceinline__ long long block_max(long long v, long long* s)
{
    const unsigned int full = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1) {
        const long long o = __shfl_xor_sync(full, v, d);
        v = o > v ? o : v;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        long long t = s[0];
        for (int w = 1; w < warps; ++w) t = s[w] > t ? s[w] : t;
        s[MAX_WARPS] = t;
    }
    __syncthreads();
    const long long result = s[MAX_WARPS];
    __syncthreads();
    return result;
}

extern "C" __global__ void kernel(const long long* __restrict__ values,
                                  const int* __restrict__ first,
                                  long long* __restrict__ out,
                                  long long n)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ long long s_red[MAX_WARPS + 1];
    __shared__ long long s_val[MAX_WARPS];
    __shared__ int s_flag[MAX_WARPS];
    __shared__ long long s_carry;

    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long b = blockIdx.x;
    const long long chunk = (n + gridDim.x - 1) / gridDim.x;
    const long long begin = b * chunk < n ? b * chunk : n;
    const long long end = begin + chunk < n ? begin + chunk : n;

    // Phase 1: maximum of the partition part that ends this block's rows.
    if (begin < end) {
        const long long f = first[end - 1];
        long long m = NEG_INF;
        for (long long i = (f > begin ? f : begin) + threadIdx.x; i < end; i += blockDim.x)
            m = values[i] > m ? values[i] : m;
        m = block_max(m, s_red);
        if (threadIdx.x == 0) out[end - 1] = m;
    }
    grid.sync();

    // Phase 2: carry from the partition that continues into this block.
    long long carry = NEG_INF;
    if (begin < end && first[begin] < begin) {
        // Every block before this one has a full chunk of rows.
        for (long long k0 = b - 1; k0 >= 0; k0 -= blockDim.x) {
            const long long k = k0 - threadIdx.x;
            long long stop = -1;
            long long v = NEG_INF;
            if (k >= 0) {
                v = out[(k + 1) * chunk - 1];
                if (first[(k + 1) * chunk - 1] >= k * chunk) stop = k;
            }
            stop = block_max(stop, s_red);
            if (k < stop) v = NEG_INF;
            v = block_max(v, s_red);
            carry = v > carry ? v : carry;
            if (stop >= 0) break;
        }
    }
    grid.sync();

    // Phase 3: segmented max scan of this block's rows.
    for (long long t0 = begin; t0 < end; t0 += blockDim.x) {
        const long long i = t0 + threadIdx.x;
        bool f = true;
        long long v = NEG_INF;
        if (i < end) {
            f = first[i] == i;
            v = values[i];
        }
        for (int d = 1; d < 32; d <<= 1) {
            const bool of = __shfl_up_sync(full, f, d);
            const long long ov = __shfl_up_sync(full, v, d);
            if (lane >= d) {
                if (!f) v = ov > v ? ov : v;
                f = f || of;
            }
        }
        if (lane == 31) {
            s_flag[warp] = f;
            s_val[warp] = v;
        }
        __syncthreads();
        // Combine with the warps before this one, then with the carry.
        for (int w = warp - 1; w >= 0 && !f; --w) {
            v = s_val[w] > v ? s_val[w] : v;
            f = s_flag[w];
        }
        if (!f) v = carry > v ? carry : v;
        if (i < end) out[i] = v;
        if (threadIdx.x == blockDim.x - 1) s_carry = v;
        __syncthreads();
        carry = s_carry;
    }
}
