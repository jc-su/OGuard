#include <cstdint>

// One block walks the column tile by tile. A thread owns 8 consecutive rows
// per tile. The number of run starts before a row is the position of its run,
// so a scan of the per-thread start counts, with a carry over the tiles, gives
// every run its output position. Work is linear for any run length.
extern "C" __global__ void kernel(const int64_t* values, int32_t* run_ends,
                                  int64_t* run_values, int32_t* count, int64_t n) {
    constexpr int ITEMS = 8;
    constexpr int TILE = 256 * ITEMS;
    __shared__ int32_t scan[256];
    const int t = threadIdx.x;
    int64_t carry = 0;
    for (int64_t tile = 0; tile < n; tile += TILE) {
        int64_t base = tile + (int64_t)t * ITEMS;
        int c = 0;
        for (int k = 0; k < ITEMS; k++) {
            int64_t i = base + k;
            if (i < n && (i == 0 || values[i] != values[i - 1])) c++;
        }
        scan[t] = c;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            int other = t >= d ? scan[t - d] : 0;
            __syncthreads();
            scan[t] += other;
            __syncthreads();
        }
        int64_t pos = carry + scan[t] - c;
        int64_t tile_total = scan[255];
        // The run that starts at row i has index pos and closes the run pos - 1.
        for (int k = 0; k < ITEMS; k++) {
            int64_t i = base + k;
            if (i < n && (i == 0 || values[i] != values[i - 1])) {
                run_values[pos] = values[i];
                if (pos > 0) run_ends[pos - 1] = (int32_t)i;
                pos++;
            }
        }
        carry += tile_total;
        __syncthreads();
    }
    // The last run ends at row n.
    if (t == 0 && n > 0) {
        run_ends[carry - 1] = (int32_t)n;
        count[0] = (int32_t)carry;
    }
}
