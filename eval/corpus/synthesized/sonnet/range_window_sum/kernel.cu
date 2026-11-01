#include <stdint.h>

// One block does the whole column in three passes over `out` because the kernel has
// no scratch memory for a cross-block carry. With P the inclusive prefix sums of
// `values`, lo(i) the first row with ts >= ts[i] - range and h(i) the last row with
// ts == ts[i]:
//   pass 0 (ascending tiles):  out[i] = P[i]
//   pass 1 (descending tiles): out[i] = P[i] - P[lo(i) - 1]   (sum of rows lo(i)..i)
//   pass 2 (ascending tiles):  out[i] = out[h(i)]             (rows of one ts share a frame)
// Every pass reads a whole tile before it writes the tile. Pass 1 reads only positions
// below the tile or inside it, pass 2 only positions inside or above it, so no value
// that is still needed has been overwritten.

#define TILE 2048

// Smallest p in [lo, hi] with ts[p] >= key, given ts[lo - 1] < key (or lo = 0) and that the
// answer is at most hi. All threads of the block call it with the same arguments; each
// round the 256 threads probe 256 positions.
__device__ int64_t lower_bound_block(const int64_t* ts, int64_t key, int64_t lo, int64_t hi)
{
    while (hi > lo) {
        const int64_t step = (hi - lo + 255) / 256;
        const int64_t q = lo + (int64_t)threadIdx.x * step;
        const int c = __syncthreads_count(q < hi && ts[q] < key);
        const int64_t new_lo = c > 0 ? lo + (int64_t)(c - 1) * step + 1 : lo;
        const int64_t reach = lo + (int64_t)c * step;
        hi = reach < hi ? reach : hi;
        lo = new_lo;
    }
    return lo;
}

// Same answer for a key whose answer is known to lie in [lo, hi]; one thread, one row.
__device__ int64_t bracket_search(const int64_t* ts, int64_t key, int64_t lo, int64_t hi)
{
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ts[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ void kernel(const int64_t* ts, const int64_t* values, int64_t* out,
                                  int64_t n, int64_t range)
{
    __shared__ uint64_t s_sum[256];
    const int t = threadIdx.x;

    // pass 0: inclusive prefix sums modulo 2^64
    uint64_t carry = 0;
    for (int64_t tile = 0; tile < n; tile += TILE) {
        const int64_t base = tile + (int64_t)t * 8;
        uint64_t sum = 0;
        for (int k = 0; k < 8; k++)
            if (base + k < n) sum += (uint64_t)values[base + k];
        s_sum[t] = sum;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            const uint64_t prev = t >= d ? s_sum[t - d] : 0;
            __syncthreads();
            s_sum[t] += prev;
            __syncthreads();
        }
        uint64_t state = carry + (t > 0 ? s_sum[t - 1] : 0);
        const uint64_t next_carry = carry + s_sum[255];
        __syncthreads();
        for (int k = 0; k < 8; k++) {
            if (base + k < n) {
                state += (uint64_t)values[base + k];
                out[base + k] = (int64_t)state;
            }
        }
        carry = next_carry;
    }
    __syncthreads();

    // pass 1: out[i] = sum of values[lo(i) .. i]
    for (int64_t tile = n > 0 ? ((n - 1) / TILE) * TILE : -1; tile >= 0; tile -= TILE) {
        const int64_t end = tile + TILE < n ? tile + TILE : n;
        const int64_t lo_first = lower_bound_block(ts, ts[tile] - range, 0, tile);
        const int64_t lo_last = lower_bound_block(ts, ts[end - 1] - range, lo_first, end - 1);
        int64_t res[8];
        for (int k = 0; k < 8; k++) {
            const int64_t i = tile + (int64_t)k * 256 + t;
            if (i < end) {
                const int64_t lo = bracket_search(ts, ts[i] - range, lo_first, lo_last);
                const uint64_t below = lo > 0 ? (uint64_t)out[lo - 1] : 0;
                res[k] = (int64_t)((uint64_t)out[i] - below);
            }
        }
        __syncthreads();
        for (int k = 0; k < 8; k++) {
            const int64_t i = tile + (int64_t)k * 256 + t;
            if (i < end) out[i] = res[k];
        }
        __syncthreads();
    }

    // pass 2: every row of one ts takes the value of the last row of that ts
    for (int64_t tile = 0; tile < n; tile += TILE) {
        const int64_t end = tile + TILE < n ? tile + TILE : n;
        const int64_t up_first = lower_bound_block(ts, ts[tile] + 1, tile + 1, n);
        const int64_t up_last =
            lower_bound_block(ts, ts[end - 1] + 1, up_first > end ? up_first : end, n);
        int64_t res[8];
        for (int k = 0; k < 8; k++) {
            const int64_t i = tile + (int64_t)k * 256 + t;
            if (i < end) {
                const int64_t up = bracket_search(ts, ts[i] + 1, up_first, up_last);
                res[k] = out[up - 1];
            }
        }
        __syncthreads();
        for (int k = 0; k < 8; k++) {
            const int64_t i = tile + (int64_t)k * 256 + t;
            if (i < end) out[i] = res[k];
        }
        __syncthreads();
    }
}
