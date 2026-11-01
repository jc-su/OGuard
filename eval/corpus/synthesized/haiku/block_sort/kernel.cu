#include <cstdint>

extern "C" __global__ void kernel(int32_t* keys, int32_t* out, int64_t n) {
    int64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int TILE_SIZE = 1024;

    if (global_id < n) {
        out[global_id] = keys[global_id];
    }

    // Assign each tile to one thread for sorting
    // Thread 0 sorts tile 0, thread 1 sorts tile 1, etc.
    int64_t tile_id = global_id;
    int64_t num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    if (tile_id < num_tiles) {
        int64_t tile_start = tile_id * TILE_SIZE;
        int64_t tile_end = (tile_start + TILE_SIZE < n) ? tile_start + TILE_SIZE : n;

        // Bubble sort within tile
        for (int64_t i = tile_start; i < tile_end; i++) {
            for (int64_t j = tile_start; j < tile_end - 1; j++) {
                if (out[j] > out[j + 1]) {
                    int32_t temp = out[j];
                    out[j] = out[j + 1];
                    out[j + 1] = temp;
                }
            }
        }
    }
}
