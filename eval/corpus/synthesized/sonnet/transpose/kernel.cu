// 32 x 32 tiles staged in shared memory (padded to avoid bank conflicts);
// a block of 256 threads moves 8 rows of the tile at a time. Grid-stride over tiles.
extern "C" __global__ void kernel(const int* source, int* out, long long rows, long long cols) {
    __shared__ int tile[32][33];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    long long tc = (cols + 31) >> 5;
    long long ntiles = ((rows + 31) >> 5) * tc;
    for (long long b = blockIdx.x; b < ntiles; b += gridDim.x) {
        long long r0 = (b / tc) * 32, c0 = (b % tc) * 32;
        for (int k = 0; k < 32; k += 8) {
            long long r = r0 + ty + k, c = c0 + tx;
            if (r < rows && c < cols) tile[ty + k][tx] = source[r * cols + c];
        }
        __syncthreads();
        for (int k = 0; k < 32; k += 8) {
            long long c = c0 + ty + k, r = r0 + tx;
            if (r < rows && c < cols) out[c * rows + r] = tile[tx][ty + k];
        }
        __syncthreads();
    }
}
