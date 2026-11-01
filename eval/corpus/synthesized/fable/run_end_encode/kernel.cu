// Each block encodes one tile of rows. A row is a run end when it is the last
// row or differs from the next one. The index of a run in the output is the
// number of run ends before it; within the tile that is a block scan, and
// across tiles it comes from count[0], which holds the number of runs written
// so far. A tile that holds run ends finds the last run end e before it, waits
// until the tile owning e has published (count[0] = P with run_ends[P - 1] =
// e + 1), writes its runs at P, P + 1, ..., and then adds its run count to
// count[0]. Tiles without run ends write nothing and wait for nothing. Blocks
// are dispatched in index order, so the tile a block waits for is always
// resident or finished.
#define ITEMS 16
#define TILE (256 * ITEMS)

extern "C" __global__ void kernel(const long long* values, int* run_ends, long long* run_values,
                                  int* count, long long n) {
    __shared__ int s_cnt[256];
    __shared__ long long s_red[256];
    __shared__ long long s_base;
    long long tid = threadIdx.x;
    long long t0 = (long long)blockIdx.x * TILE;
    if (t0 >= n) return;
    long long t1 = t0 + TILE < n ? t0 + TILE : n;
    long long first = t0 + tid * ITEMS;
    int cnt = 0;
    for (int k = 0; k < ITEMS; ++k) {
        long long i = first + k;
        if (i < t1 && (i == n - 1 || values[i] != values[i + 1])) ++cnt;
    }
    s_cnt[tid] = cnt;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {
        int v = tid >= d ? s_cnt[tid - d] : 0;
        __syncthreads();
        s_cnt[tid] += v;
        __syncthreads();
    }
    long long excl = s_cnt[tid] - cnt;
    int total = s_cnt[255];
    if (total == 0) return;
    // Last run end before the tile.
    long long e = -1;
    for (long long b = t0; b > 0 && e < 0;) {
        long long a = b - TILE > 0 ? b - TILE : 0;
        long long best = -1;
        for (long long i = a + tid; i < b; i += 256) if (values[i] != values[i + 1]) best = i;
        s_red[tid] = best;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s && s_red[tid + s] > s_red[tid]) s_red[tid] = s_red[tid + s];
            __syncthreads();
        }
        e = s_red[0];
        __syncthreads();
        b = a;
    }
    if (tid == 0) {
        long long base = 0;
        if (e >= 0) {
            volatile int* vcount = count;
            volatile int* vends = run_ends;
            while (true) {
                long long p = *vcount;
                if (p > 0 && (long long)vends[p - 1] == e + 1) { base = p; break; }
            }
        }
        s_base = base;
    }
    __syncthreads();
    long long j = s_base + excl;
    for (int k = 0; k < ITEMS; ++k) {
        long long i = first + k;
        if (i < t1 && (i == n - 1 || values[i] != values[i + 1])) {
            run_ends[j] = (int)(i + 1);
            run_values[j] = values[i];
            ++j;
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(count, total);
}
