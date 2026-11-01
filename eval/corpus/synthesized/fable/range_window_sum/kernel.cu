// Each block handles one tile of rows. Every row's frame [lo, hi] comes from
// two binary searches over ts. The frame is split into the part inside the
// tile, the part before it, and the part after it; each part is read in
// chunks of TILE rows whose prefix sums are built in shared memory, so a
// frame of any length costs one coalesced pass over it per block, and no
// global scratch is needed. Sums use unsigned arithmetic so that they wrap.
#define ITEMS 8
#define TILE (256 * ITEMS)

// s_pref[k] = sum of src[a : a + k] for k in [0, b - a], b - a <= TILE.
// Every thread of the block must call this; it synchronizes inside.
__device__ static void chunk_prefix(const long long* src, long long a, long long b,
                                    unsigned long long* s_pref, unsigned long long* s_scan, long long tid) {
    long long base = a + tid * ITEMS;
    unsigned long long local = 0;
    for (int k = 0; k < ITEMS; ++k) if (base + k < b) local += (unsigned long long)src[base + k];
    s_scan[tid] = local;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {
        unsigned long long v = tid >= d ? s_scan[tid - d] : 0;
        __syncthreads();
        s_scan[tid] += v;
        __syncthreads();
    }
    unsigned long long running = s_scan[tid] - local;
    if (tid == 0) s_pref[0] = 0;
    for (int k = 0; k < ITEMS; ++k) {
        if (base + k < b) { running += (unsigned long long)src[base + k]; s_pref[tid * ITEMS + k + 1] = running; }
    }
    __syncthreads();
}

__device__ static long long lower_bound(const long long* ts, long long n, long long key) {
    long long lo = 0, hi = n;
    while (lo < hi) { long long mid = (lo + hi) >> 1; if (ts[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

__device__ static long long upper_bound(const long long* ts, long long n, long long key) {
    long long lo = 0, hi = n;
    while (lo < hi) { long long mid = (lo + hi) >> 1; if (ts[mid] <= key) lo = mid + 1; else hi = mid; }
    return lo;
}

extern "C" __global__ void kernel(const long long* ts, const long long* values, long long* out,
                                  long long n, long long range) {
    __shared__ unsigned long long s_pref[TILE + 1];
    __shared__ unsigned long long s_scan[256];
    __shared__ long long s_bound[2];
    long long tid = threadIdx.x;
    long long t0 = (long long)blockIdx.x * TILE;
    if (t0 >= n) return;
    long long t1 = t0 + TILE < n ? t0 + TILE : n;
    long long base = t0 + tid * ITEMS;
    long long lo[ITEMS], hi[ITEMS];
    unsigned long long acc[ITEMS];
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        lo[k] = i; hi[k] = i - 1; acc[k] = 0;
        if (i < t1) {
            long long x = ts[i];
            lo[k] = lower_bound(ts, n, x - range);
            hi[k] = upper_bound(ts, n, x) - 1;
        }
    }
    if (tid == 0) s_bound[0] = lo[0];
    long long last = t1 - 1 - t0;
    if (tid == last / ITEMS) s_bound[1] = hi[last - tid * ITEMS];
    // Part inside the tile.
    chunk_prefix(values, t0, t1, s_pref, s_scan, tid);
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1) {
            long long j0 = lo[k] > t0 ? lo[k] : t0;
            long long j1 = hi[k] < t1 - 1 ? hi[k] : t1 - 1;
            acc[k] += s_pref[j1 - t0 + 1] - s_pref[j0 - t0];
        }
    }
    __syncthreads();
    long long first_lo = s_bound[0], last_hi = s_bound[1];
    // Part before the tile, in chunks going backward.
    for (long long b = t0; b > first_lo;) {
        long long a = b - TILE > first_lo ? b - TILE : first_lo;
        chunk_prefix(values, a, b, s_pref, s_scan, tid);
        for (int k = 0; k < ITEMS; ++k) {
            if (base + k < t1 && lo[k] < b) acc[k] += lo[k] >= a ? s_pref[b - a] - s_pref[lo[k] - a] : s_pref[b - a];
        }
        __syncthreads();
        b = a;
    }
    // Part after the tile, in chunks going forward.
    for (long long a = t1; a <= last_hi;) {
        long long b = a + TILE < last_hi + 1 ? a + TILE : last_hi + 1;
        chunk_prefix(values, a, b, s_pref, s_scan, tid);
        for (int k = 0; k < ITEMS; ++k) {
            if (base + k < t1 && hi[k] >= a) acc[k] += hi[k] < b ? s_pref[hi[k] - a + 1] : s_pref[b - a];
        }
        __syncthreads();
        a = b;
    }
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + k;
        if (i < t1) out[i] = (long long)acc[k];
    }
}
