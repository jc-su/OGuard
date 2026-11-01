// Each block scans one tile of rows. When the partition of the first tile row
// starts before the tile, the block first reduces the values from that
// partition start to the tile start (coalesced, one block reduction) to get
// the carry. The tile itself is a segmented block scan: each thread scans its
// own consecutive rows, a shared-memory scan combines the thread aggregates,
// and each thread rescans its rows with its prefix.
#define ITEMS 8
#define TILE (256 * ITEMS)
#define INT64_MIN_V (-0x7FFFFFFFFFFFFFFFll - 1)

extern "C" __global__ void kernel(const long long* values, const int* first, long long* out, long long n) {
    __shared__ int s_flag[256];
    __shared__ long long s_val[256];
    long long tid = threadIdx.x;
    long long t0 = (long long)blockIdx.x * TILE;
    if (t0 >= n) return;
    long long t1 = t0 + TILE < n ? t0 + TILE : n;
    long long carry = INT64_MIN_V;
    long long seg = first[t0];
    if (seg < t0) {
        long long best = INT64_MIN_V;
        for (long long i = seg + tid; i < t0; i += 256) { long long v = values[i]; if (v > best) best = v; }
        s_val[tid] = best;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s && s_val[tid + s] > s_val[tid]) s_val[tid] = s_val[tid + s];
            __syncthreads();
        }
        carry = s_val[0];
        __syncthreads();
    }
    long long base = t0 + tid * ITEMS;
    int f = 0;
    long long agg = INT64_MIN_V;
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1) {
            long long v = values[i];
            if ((long long)first[i] == i) { f = 1; agg = v; }
            else if (v > agg) agg = v;
        }
    }
    s_flag[tid] = f; s_val[tid] = agg;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {
        int pf = 0; long long pv = INT64_MIN_V;
        if (tid >= d) { pf = s_flag[tid - d]; pv = s_val[tid - d]; }
        __syncthreads();
        if (tid >= d && s_flag[tid] == 0) { s_flag[tid] = pf; if (pv > s_val[tid]) s_val[tid] = pv; }
        __syncthreads();
    }
    long long running = carry;
    if (tid > 0) {
        if (s_flag[tid - 1] != 0) running = s_val[tid - 1];
        else if (s_val[tid - 1] > running) running = s_val[tid - 1];
    }
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1) {
            long long v = values[i];
            if ((long long)first[i] == i || v > running) running = v;
            out[i] = running;
        }
    }
}
