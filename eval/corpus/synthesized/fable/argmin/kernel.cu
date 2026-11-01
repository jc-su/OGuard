// Each block reduces a tile to its (minimum, smallest row) pair, then thread 0
// merges the pair into row[0] with a compare-and-swap loop that keeps the pair
// with the smaller value, or the smaller row on a tie.
#define TILE 4096

extern "C" __global__ void kernel(const long long* values, int* row, long long n) {
    __shared__ long long s_val[256];
    __shared__ long long s_row[256];
    long long tid = threadIdx.x;
    long long start = (long long)blockIdx.x * TILE;
    long long end = start + TILE < n ? start + TILE : n;
    long long bv = 0, br = -1;
    for (long long i = start + tid; i < end; i += 256) {
        long long v = values[i];
        if (br < 0 || v < bv) { bv = v; br = i; }
    }
    s_val[tid] = bv; s_row[tid] = br;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            long long ov = s_val[tid + s], orow = s_row[tid + s];
            if (orow >= 0 && (br < 0 || ov < bv || (ov == bv && orow < br))) {
                bv = ov; br = orow; s_val[tid] = bv; s_row[tid] = br;
            }
        }
        __syncthreads();
    }
    if (tid == 0 && br >= 0) {
        int cur = atomicAdd(row, 0);
        while (true) {
            if (cur >= 0) {
                long long cv = values[cur];
                if (cv < bv || (cv == bv && cur < br)) break;
            }
            int old = atomicCAS(row, cur, (int)br);
            if (old == cur) break;
            cur = old;
        }
    }
}
