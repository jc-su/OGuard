// Transpose: out[j * rows + i] = source[i * cols + j] for a row-major matrix.
#include <cstdint>

constexpr int kTile = 32;

extern "C" __global__ void kernel(const int32_t* source, int32_t* out, int64_t rows,
                                  int64_t cols) {
    // Each block moves 32 x 32 tiles through shared memory so that both the reads of
    // source and the writes of out are coalesced. The extra column keeps the column
    // reads of the tile free of bank conflicts.
    __shared__ int32_t tile[kTile][kTile + 1];
    const int tx = threadIdx.x % kTile;
    const int ty = threadIdx.x / kTile;
    const int step = blockDim.x / kTile;
    const int64_t tiles_c = (cols + kTile - 1) / kTile;
    const int64_t ntiles = ((rows + kTile - 1) / kTile) * tiles_c;
    for (int64_t b = blockIdx.x; b < ntiles; b += gridDim.x) {
        const int64_t r0 = (b / tiles_c) * kTile;
        const int64_t c0 = (b % tiles_c) * kTile;
        const int64_t c = c0 + tx;
        for (int y = ty; y < kTile; y += step) {
            const int64_t r = r0 + y;
            if (r < rows && c < cols) tile[y][tx] = source[r * cols + c];
        }
        __syncthreads();
        const int64_t r = r0 + tx;
        for (int y = ty; y < kTile; y += step) {
            const int64_t col = c0 + y;
            if (col < cols && r < rows) out[col * rows + r] = tile[tx][y];
        }
        // The tile is written again in the next iteration.
        __syncthreads();
    }
}
