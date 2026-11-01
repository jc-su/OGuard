// fill_forward: out[i] = values[j] for the largest valid row j <= i, or 0 when rows
// 0..i are all null.
//
// A run of nulls can span the whole column, so the kernel is a cooperative launch
// (all blocks resident, grid.sync between phases) and block b owns rows
// [begin, end) = [b * chunk, (b + 1) * chunk):
//   1. Each block finds its last valid row, scanning the validity bits backward from
//      the end of its rows 256 rows at a time, and stores that row number (or -1)
//      in out[end - 1] (overwritten in phase 3).
//   2. Each block looks back over the previous blocks, 256 at a time; the nearest
//      block with a valid row holds the last valid row before this block, and its
//      value is the carry (0 when there is none).
//   3. Each block scans its rows in tiles of 256, keeping the value of the last
//      valid row (warp shuffles, then warp totals in shared memory), uses the carry
//      for rows before the first valid row of the tile, and writes out.

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MAX_WARPS 32

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

__device__ __forceinline__ bool is_valid(const unsigned char* __restrict__ valid, long long i)
{
    return (valid[i >> 3] >> (i & 7)) & 1;
}

extern "C" __global__ void kernel(const long long* __restrict__ values,
                                  const unsigned char* __restrict__ valid,
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

    // Phase 1: last valid row of this block's rows.
    if (begin < end) {
        long long last = -1;
        for (long long t1 = end; t1 > begin && last < 0; t1 -= blockDim.x) {
            const long long i = t1 - 1 - threadIdx.x;
            last = block_max(i >= begin && is_valid(valid, i) ? i : -1, s_red);
        }
        if (threadIdx.x == 0) out[end - 1] = last;
    }
    grid.sync();

    // Phase 2: value of the last valid row before this block.
    long long carry = 0;
    if (begin < end) {
        // Every block before this one has a full chunk of rows, and later blocks
        // have later rows, so the largest stored row is the nearest one.
        for (long long k0 = b - 1; k0 >= 0; k0 -= blockDim.x) {
            const long long k = k0 - threadIdx.x;
            const long long last = block_max(k >= 0 ? out[(k + 1) * chunk - 1] : -1, s_red);
            if (last >= 0) {
                carry = values[last];
                break;
            }
        }
    }
    grid.sync();

    // Phase 3: forward fill of this block's rows.
    for (long long t0 = begin; t0 < end; t0 += blockDim.x) {
        const long long i = t0 + threadIdx.x;
        bool f = i < end && is_valid(valid, i);
        long long v = f ? values[i] : 0;
        for (int d = 1; d < 32; d <<= 1) {
            const bool of = __shfl_up_sync(full, f, d);
            const long long ov = __shfl_up_sync(full, v, d);
            if (lane >= d && !f) {
                v = ov;
                f = of;
            }
        }
        if (lane == 31) {
            s_flag[warp] = f;
            s_val[warp] = v;
        }
        __syncthreads();
        // Take the last valid value of the warps before this one, then the carry.
        for (int w = warp - 1; w >= 0 && !f; --w) {
            if (s_flag[w]) {
                v = s_val[w];
                f = true;
            }
        }
        if (!f) v = carry;
        if (i < end) out[i] = v;
        if (threadIdx.x == blockDim.x - 1) s_carry = v;
        __syncthreads();
        carry = s_carry;
    }
}
