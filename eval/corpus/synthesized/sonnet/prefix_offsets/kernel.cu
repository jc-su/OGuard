// Block 0 scans the whole column in tiles of 256 * 8 elements, carrying the
// running total between tiles. Other blocks return, so the result does not
// depend on the grid size. No scratch space is available in the signature.
// Needs a block size of 256.
extern "C" __global__ void kernel(const int* lengths, int* offsets, long long n) {
    if (blockIdx.x != 0) return;
    __shared__ int wsum[8];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) offsets[0] = 0;
    int carry = 0;
    for (long long base = 0; base < n; base += 2048) {
        long long first = base + tid * 8;
        int local[8], s = 0;
        for (int k = 0; k < 8; k++) {
            local[k] = (first + k < n) ? lengths[first + k] : 0;
            s += local[k];
        }
        int x = s;
        for (int d = 1; d < 32; d <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) wsum[warp] = x;
        __syncthreads();
        int wpre = 0, total = 0;
        for (int w = 0; w < 8; w++) {
            if (w < warp) wpre += wsum[w];
            total += wsum[w];
        }
        int run = carry + wpre + x - s;
        for (int k = 0; k < 8; k++) {
            run += local[k];
            if (first + k < n) offsets[first + k + 1] = run;
        }
        carry += total;
        __syncthreads();
    }
}
