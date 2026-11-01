#include <cstdint>

extern "C" __global__ void kernel(int32_t* values, int32_t* out, int32_t* totals, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int TILE_SIZE = 1024;

    if (idx < n) {
        int64_t tile_idx = idx / TILE_SIZE;
        int64_t local_idx = idx % TILE_SIZE;
        int64_t tile_start = tile_idx * TILE_SIZE;
        int64_t tile_end = min(tile_start + TILE_SIZE, n);

        // Compute prefix sum within the tile
        int64_t prefix_sum = 0;
        for (int64_t i = tile_start; i < tile_start + local_idx; i++) {
            prefix_sum += values[i];
        }
        out[idx] = (int32_t)prefix_sum;

        // Compute and store tile total (by first thread of tile)
        if (local_idx == 0) {
            int64_t total = 0;
            for (int64_t i = tile_start; i < tile_end; i++) {
                total += values[i];
            }
            totals[tile_idx] = (int32_t)total;
        }
    }
}
