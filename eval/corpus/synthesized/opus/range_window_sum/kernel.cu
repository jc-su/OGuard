// range_window_sum: out[i] = sum of values[j] over the rows j with
// ts[i] - range <= ts[j] <= ts[i] (RANGE BETWEEN range PRECEDING AND CURRENT ROW),
// modulo 2^64.
//
// With P(q) = values[0] + ... + values[q - 1], out[i] = P(U_i) - P(L_i), where
// L_i = first row with ts >= ts[i] - range and U_i = first row with ts > ts[i]. L_i and
// U_i are found by galloping search from row i (cost logarithmic in the distance).
// A window can cover the whole column, so the kernel is a cooperative launch (all
// blocks resident, grid.sync between phases) and block b owns rows
// [begin, end) = [b * chunk, (b + 1) * chunk):
//   1. Each block sums its values and stores the sum in out[begin].
//   2. Each block copies all block sums to shared memory and turns them into
//      exclusive prefix sums, so P(k * chunk) is known for every block k. A
//      grid.sync then lets phase 3 overwrite the stored sums.
//   3. Each block evaluates P at the queries L_i and U_i of its rows, 256 rows per
//      tile, with two cursors (position, P(position)). L_i and U_i are nondecreasing
//      in i, so each cursor only moves forward: it adds values 256 at a time with a
//      block-wide inclusive scan and each thread picks P at its query from the scan.
//      The U cursor starts at P(begin); the L cursor starts at the block boundary
//      below L_begin, so it reads at most one chunk before reaching it. In total a
//      block reads O(chunk + its window span) values.

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MAX_BLOCKS 1024
#define MAX_THREADS 1024
#define MAX_WARPS 32

typedef unsigned long long u64;

// Inclusive sum over the block; total receives the block total.
__device__ __forceinline__ u64 block_inclusive_sum(u64 v, u64* s_warp, u64& total)
{
    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int d = 1; d < 32; d <<= 1) {
        const u64 o = __shfl_up_sync(full, v, d);
        if (lane >= d) v += o;
    }
    if (lane == 31) s_warp[warp] = v;
    __syncthreads();
    u64 before = 0, all = 0;
    for (int w = 0; w < warps; ++w) {
        if (w < warp) before += s_warp[w];
        all += s_warp[w];
    }
    __syncthreads();
    total = all;
    return v + before;
}

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
    long long m = s[0];
    for (int w = 1; w < warps; ++w) m = s[w] > m ? s[w] : m;
    __syncthreads();
    return m;
}

// First j in [0, i] with ts[j] >= x, given ts[i] >= x.
__device__ __forceinline__ long long lower_bound_back(const long long* __restrict__ ts, long long i, long long x)
{
    long long hi = i, lo, step = 1;
    while (true) {
        lo = hi - step;
        if (lo < 0) {
            lo = -1;
            break;
        }
        if (ts[lo] < x) break;
        hi = lo;
        step <<= 1;
    }
    for (lo += 1; lo < hi;) {  // answer in [lo, hi]
        const long long mid = lo + ((hi - lo) >> 1);
        if (ts[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// First j in (i, n] with ts[j] > x (j = n when none), given ts[i] <= x.
__device__ __forceinline__ long long upper_bound_fwd(const long long* __restrict__ ts, long long i, long long n,
                                                     long long x)
{
    long long lo = i, hi, step = 1;
    while (true) {
        hi = lo + step;
        if (hi >= n) {
            hi = n;
            break;
        }
        if (ts[hi] > x) break;
        lo = hi;
        step <<= 1;
    }
    for (lo += 1; lo < hi;) {  // answer in [lo, hi]
        const long long mid = lo + ((hi - lo) >> 1);
        if (ts[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Moves the cursor (pos, val = P(pos)) forward to q_max, the largest query of the
// block (block-uniform), and returns P(q) for this thread's query q in [pos, q_max].
__device__ __forceinline__ u64 prefix_at(const long long* __restrict__ values, long long& pos, u64& val,
                                         long long q, long long q_max, u64* s_warp, u64* s_scan)
{
    u64 result = val;
    while (pos < q_max) {
        const long long step = q_max - pos < blockDim.x ? q_max - pos : blockDim.x;
        const u64 x = threadIdx.x < step ? (u64)values[pos + threadIdx.x] : 0;
        u64 total;
        s_scan[threadIdx.x] = block_inclusive_sum(x, s_warp, total);
        __syncthreads();
        if (q > pos && q <= pos + step) result = val + s_scan[q - pos - 1];
        __syncthreads();
        pos += step;
        val += total;
    }
    return result;
}

extern "C" __global__ void kernel(const long long* __restrict__ ts,
                                  const long long* __restrict__ values,
                                  long long* __restrict__ out,
                                  long long n, long long range)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ u64 s_prefix[MAX_BLOCKS];
    __shared__ u64 s_scan[MAX_THREADS];
    __shared__ u64 s_warp[MAX_WARPS];
    __shared__ long long s_max[MAX_WARPS];
    __shared__ long long s_low;

    if (gridDim.x > MAX_BLOCKS || blockDim.x > MAX_THREADS) __trap();
    const long long blocks = gridDim.x;
    const long long b = blockIdx.x;
    const long long chunk = (n + blocks - 1) / blocks;
    const long long begin = b * chunk < n ? b * chunk : n;
    const long long end = begin + chunk < n ? begin + chunk : n;

    // Phase 1: sum of this block's values, stored in out[begin].
    {
        u64 s = 0;
        for (long long i = begin + threadIdx.x; i < end; i += blockDim.x) s += (u64)values[i];
        u64 total;
        block_inclusive_sum(s, s_warp, total);
        if (threadIdx.x == 0 && begin < end) out[begin] = (long long)total;
    }
    grid.sync();

    // Phase 2: s_prefix[k] = P(k * chunk) for every block k.
    {
        const long long per = (blocks + blockDim.x - 1) / blockDim.x;
        u64 local = 0;
        for (long long e = 0; e < per; ++e) {
            const long long k = threadIdx.x * per + e;
            if (k < blocks) {
                s_prefix[k] = k * chunk < n ? (u64)out[k * chunk] : 0;
                local += s_prefix[k];
            }
        }
        u64 total;
        u64 running = block_inclusive_sum(local, s_warp, total) - local;
        for (long long e = 0; e < per; ++e) {
            const long long k = threadIdx.x * per + e;
            if (k < blocks) {
                const u64 v = s_prefix[k];
                s_prefix[k] = running;
                running += v;
            }
        }
        if (threadIdx.x == 0 && begin < end) s_low = lower_bound_back(ts, begin, ts[begin] - range);
        __syncthreads();
    }
    grid.sync();

    // Phase 3: out[i] = P(U_i) - P(L_i).
    if (begin < end) {
        long long u_pos = begin;
        u64 u_val = s_prefix[b];
        long long l_pos = (s_low / chunk) * chunk;
        u64 l_val = s_prefix[s_low / chunk];
        for (long long t0 = begin; t0 < end; t0 += blockDim.x) {
            const long long i = t0 + threadIdx.x;
            const bool active = i < end;
            long long lq = -1, uq = -1;
            if (active) {
                const long long x = ts[i];
                lq = lower_bound_back(ts, i, x - range);
                uq = upper_bound_fwd(ts, i, n, x);
            }
            const long long lq_max = block_max(lq, s_max);
            const long long uq_max = block_max(uq, s_max);
            if (!active) {
                lq = l_pos;
                uq = u_pos;
            }
            const u64 pl = prefix_at(values, l_pos, l_val, lq, lq_max, s_warp, s_scan);
            const u64 pu = prefix_at(values, u_pos, u_val, uq, uq_max, s_warp, s_scan);
            if (active) out[i] = (long long)(pu - pl);
        }
    }
}
