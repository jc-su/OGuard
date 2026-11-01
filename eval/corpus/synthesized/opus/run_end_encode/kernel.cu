// run_end_encode: Arrow run-end encoding of values. count[0] = number of runs r; for
// run j, run_ends[j] = row after its last row and run_values[j] = its value.
//
// Row i ends a run when i == n - 1 or values[i] != values[i + 1]; run j is the run
// ending at the (j + 1)-th such row. The kernel is a cooperative launch (all blocks
// resident, grid.sync between phases) and block b owns rows [b * chunk, (b + 1) * chunk):
//   1. Each block counts the run ends in its rows. A block with c > 0 run ends takes
//      a slot from the counter count[0] (zero on entry) and records (b, c) there in
//      run_ends / run_values. At most min(blocks, r) slots are used, all below r, so
//      the positions at r and above are never touched.
//   2. Each block reads the recorded slots and sums the counts of the blocks before
//      it (its first output position) and of all blocks (r).
//   3. Block 0 sets count[0] = r, and each block writes its runs at its first output
//      position plus their rank among the run ends of its rows (a block-wide scan of
//      ballots per tile of 256 rows).

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MAX_WARPS 32

__device__ __forceinline__ bool is_run_end(const long long* __restrict__ values, long long i, long long n)
{
    return i == n - 1 || values[i] != values[i + 1];
}

// Sum of v over the block; valid in every thread. s holds MAX_WARPS + 1 entries.
__device__ __forceinline__ long long block_sum(long long v, long long* s)
{
    const unsigned int full = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(full, v, d);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        long long t = 0;
        for (int w = 0; w < warps; ++w) t += s[w];
        s[MAX_WARPS] = t;
    }
    __syncthreads();
    const long long total = s[MAX_WARPS];
    __syncthreads();
    return total;
}

extern "C" __global__ void kernel(const long long* __restrict__ values,
                                  int* __restrict__ run_ends,
                                  long long* __restrict__ run_values,
                                  int* __restrict__ count,
                                  long long n)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ long long s_red[MAX_WARPS + 1];
    __shared__ int s_warp[MAX_WARPS];

    const long long b = blockIdx.x;
    const long long chunk = (n + gridDim.x - 1) / gridDim.x;
    const long long begin = b * chunk < n ? b * chunk : n;
    const long long end = begin + chunk < n ? begin + chunk : n;

    // Phase 1: run ends in [begin, end).
    long long c = 0;
    for (long long i = begin + threadIdx.x; i < end; i += blockDim.x) c += is_run_end(values, i, n);
    c = block_sum(c, s_red);
    if (threadIdx.x == 0 && c > 0) {
        const int slot = atomicAdd(count, 1);
        run_ends[slot] = (int)b;
        run_values[slot] = c;
    }
    grid.sync();

    // Phase 2: run ends before this block, and in total.
    const long long used = count[0];
    long long before = 0, total = 0;
    for (long long j = threadIdx.x; j < used; j += blockDim.x) {
        const long long cj = run_values[j];
        total += cj;
        if (run_ends[j] < b) before += cj;
    }
    before = block_sum(before, s_red);
    total = block_sum(total, s_red);
    grid.sync();

    // Phase 3: write the runs.
    if (b == 0 && threadIdx.x == 0) count[0] = (int)total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    long long pos = before;
    for (long long t0 = begin; t0 < end; t0 += blockDim.x) {
        const long long i = t0 + threadIdx.x;
        const bool flag = i < end && is_run_end(values, i, n);
        const unsigned int ballot = __ballot_sync(0xffffffffu, flag);
        if (lane == 0) s_warp[warp] = __popc(ballot);
        __syncthreads();
        int rank = __popc(ballot & ((1u << lane) - 1u));
        int tile_total = 0;
        for (int w = 0; w < warps; ++w) {
            if (w < warp) rank += s_warp[w];
            tile_total += s_warp[w];
        }
        if (flag) {
            run_ends[pos + rank] = (int)(i + 1);
            run_values[pos + rank] = values[i];
        }
        pos += tile_total;
        __syncthreads();
    }
}
