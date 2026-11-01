#include <cstdint>

// One block walks the column tile by tile. A thread owns 8 consecutive rows
// per tile. The state of a row is the pair (partition started, running max),
// combined by (f1, v1) . (f2, v2) = (f1 | f2, f2 ? v2 : max(v1, v2)), which is
// associative, so a scan of the per-thread aggregates, with a carry over the
// tiles, gives every thread its starting state. Work is linear for any
// partition length.
__device__ __forceinline__ void combine(int& f1, int64_t& v1, int f2, int64_t v2) {
    v1 = f2 ? v2 : (v1 > v2 ? v1 : v2);
    f1 |= f2;
}

extern "C" __global__ void kernel(const int64_t* values, const int32_t* first,
                                  int64_t* out, int64_t n) {
    constexpr int ITEMS = 8;
    constexpr int TILE = 256 * ITEMS;
    __shared__ int scan_f[256];
    __shared__ int64_t scan_v[256];
    const int t = threadIdx.x;
    int carry_f = 0;
    int64_t carry_v = INT64_MIN;
    for (int64_t tile = 0; tile < n; tile += TILE) {
        int64_t base = tile + (int64_t)t * ITEMS;
        int f = 0;
        int64_t v = INT64_MIN;
        for (int k = 0; k < ITEMS; k++) {
            int64_t i = base + k;
            if (i < n) combine(f, v, first[i] == i, values[i]);
        }
        scan_f[t] = f;
        scan_v[t] = v;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            int of = 0;
            int64_t ov = INT64_MIN;
            if (t >= d) { of = scan_f[t - d]; ov = scan_v[t - d]; }
            __syncthreads();
            combine(of, ov, scan_f[t], scan_v[t]);
            scan_f[t] = of;
            scan_v[t] = ov;
            __syncthreads();
        }
        // State before this thread's slice: carry . inclusive state of thread t - 1.
        int sf = carry_f;
        int64_t sv = carry_v;
        if (t > 0) combine(sf, sv, scan_f[t - 1], scan_v[t - 1]);
        int tile_f = scan_f[255];
        int64_t tile_v = scan_v[255];
        for (int k = 0; k < ITEMS; k++) {
            int64_t i = base + k;
            if (i < n) {
                combine(sf, sv, first[i] == i, values[i]);
                out[i] = sv;
            }
        }
        combine(carry_f, carry_v, tile_f, tile_v);
        __syncthreads();
    }
}
