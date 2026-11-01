// block_sort: write the keys of each tile of 1024 rows to the same rows of
// out, sorted ascending within the tile.
// One block of 256 threads sorts one tile with a bitonic sort in shared
// memory. A short last tile is padded with INT_MAX, which sorts after every
// real key. Blocks past the last tile return, so a larger grid is also correct.
#include <climits>

extern "C" __global__ void kernel(const int* keys,
                                  int* out,
                                  long long n) {
    const int TILE = 1024;
    const int PAIRS = TILE / 2;
    long long tile = blockIdx.x;
    if (tile >= (n + TILE - 1) / TILE) return;
    int tid = threadIdx.x;
    long long base = tile * TILE;

    __shared__ int buf[TILE];
    for (int j = tid; j < TILE; j += blockDim.x) {
        long long r = base + j;
        buf[j] = (r < n) ? keys[r] : INT_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= TILE; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            for (int p = tid; p < PAIRS; p += blockDim.x) {
                int i = (p / j) * (j << 1) + (p & (j - 1));
                int partner = i | j;
                int x = buf[i];
                int y = buf[partner];
                if (((i & k) == 0) == (x > y)) {
                    buf[i] = y;
                    buf[partner] = x;
                }
            }
            __syncthreads();
        }
    }

    for (int j = tid; j < TILE; j += blockDim.x) {
        long long r = base + j;
        if (r < n) out[r] = buf[j];
    }
}
