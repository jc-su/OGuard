// Each block fills one tile of rows. The carry into the tile is the last valid
// row before it, found by scanning the validity bitmap backward 256 bytes at a
// time with a block reduction; a null run of any length costs one coalesced
// pass over its bitmap bytes per block. The tile itself is a block scan with
// the operator "take the right side when it is valid": each thread scans its
// own consecutive rows, a shared-memory scan combines the thread aggregates,
// and each thread rescans its rows.
#define ITEMS 8
#define TILE (256 * ITEMS)

extern "C" __global__ void kernel(const long long* values, const unsigned char* valid, long long* out, long long n) {
    __shared__ int s_flag[256];
    __shared__ long long s_val[256];
    __shared__ long long s_red[256];
    long long tid = threadIdx.x;
    long long t0 = (long long)blockIdx.x * TILE;
    if (t0 >= n) return;
    long long t1 = t0 + TILE < n ? t0 + TILE : n;
    long long carry_row = -1;
    if (t0 > 0) {
        long long top = (t0 - 1) >> 3;
        for (long long q_hi = top; q_hi >= 0 && carry_row < 0; q_hi -= 256) {
            long long q = q_hi - tid;
            long long best = -1;
            if (q >= 0) {
                unsigned int byte = valid[q];
                if (q == top) byte &= (1u << (t0 - q * 8)) - 1u;
                if (byte != 0) best = q * 8 + (31 - __clz(byte));
            }
            s_red[tid] = best;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s && s_red[tid + s] > s_red[tid]) s_red[tid] = s_red[tid + s];
                __syncthreads();
            }
            carry_row = s_red[0];
            __syncthreads();
        }
    }
    long long carry = carry_row >= 0 ? values[carry_row] : 0;
    long long base = t0 + tid * ITEMS;
    int f = 0;
    long long agg = 0;
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1 && ((valid[i >> 3] >> (i & 7)) & 1)) { f = 1; agg = values[i]; }
    }
    s_flag[tid] = f; s_val[tid] = agg;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {
        int pf = 0; long long pv = 0;
        if (tid >= d) { pf = s_flag[tid - d]; pv = s_val[tid - d]; }
        __syncthreads();
        if (tid >= d && s_flag[tid] == 0) { s_flag[tid] = pf; s_val[tid] = pv; }
        __syncthreads();
    }
    long long running = carry;
    if (tid > 0 && s_flag[tid - 1] != 0) running = s_val[tid - 1];
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1) {
            if ((valid[i >> 3] >> (i & 7)) & 1) running = values[i];
            out[i] = running;
        }
    }
}
