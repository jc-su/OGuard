// tile_scan: out[i] is the exclusive prefix sum of values within row i's tile
// of 1024 rows, and totals[t] is the sum of tile t.
// One block of 256 threads handles one tile; each thread holds 4 consecutive
// rows. Blocks past the last tile return, so a larger grid is also correct.
extern "C" __global__ void kernel(const int* values,
                                  int* out,
                                  int* totals,
                                  long long n) {
    const int TILE = 1024;
    const int WARPS = 8;
    long long tile = blockIdx.x;
    if (tile >= (n + TILE - 1) / TILE) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    long long base = tile * TILE + (long long)tid * 4;

    int v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = (base + j < n) ? values[base + j] : 0;
    int local = v[0] + v[1] + v[2] + v[3];

    // Inclusive scan of the per-thread sums within each warp.
    int x = local;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int y = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= d) x += y;
    }

    // Inclusive scan of the warp totals by warp 0.
    __shared__ int warp_sums[WARPS];
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = (lane < WARPS) ? warp_sums[lane] : 0;
#pragma unroll
        for (int d = 1; d < WARPS; d <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFFu, w, d);
            if (lane >= d) w += y;
        }
        if (lane < WARPS) warp_sums[lane] = w;
    }
    __syncthreads();

    int prefix = x - local + (warp > 0 ? warp_sums[warp - 1] : 0);
    if (base < n) out[base] = prefix;
    if (base + 1 < n) out[base + 1] = prefix + v[0];
    if (base + 2 < n) out[base + 2] = prefix + v[0] + v[1];
    if (base + 3 < n) out[base + 3] = prefix + v[0] + v[1] + v[2];
    if (tid == blockDim.x - 1) totals[tile] = prefix + local;
}
