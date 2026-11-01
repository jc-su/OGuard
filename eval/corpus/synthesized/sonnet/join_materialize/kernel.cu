#include <stdint.h>

// One thread per probe row finds the first matching build row by binary search. A row with
// at most 32 matches writes them itself. The lanes of the warp then write the matches of each
// remaining row together, one row at a time, so a row with many matches does not stall its warp
// on a single thread. Every lane reaches the ballot; only the work is guarded by i < p.
extern "C" __global__ void kernel(const int64_t* __restrict__ build,
                                  const int64_t* __restrict__ probe,
                                  const int32_t* __restrict__ offsets,
                                  int32_t* __restrict__ out_build,
                                  int32_t* __restrict__ out_probe, int64_t b, int64_t p) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    const int lane = threadIdx.x & 31;

    int32_t base = 0, count = 0, first = 0;
    if (i < p) {
        base = offsets[i];
        count = offsets[i + 1] - base;
        if (count > 0) {
            const int64_t key = probe[i];
            int64_t lo = 0, hi = b;  // first build row not less than key
            while (lo < hi) {
                int64_t mid = lo + ((hi - lo) >> 1);
                if (build[mid] < key) lo = mid + 1; else hi = mid;
            }
            first = (int32_t)lo;
        }
    }

    if (count <= 32) {
        for (int32_t j = 0; j < count; j++) {
            out_build[base + j] = first + j;
            out_probe[base + j] = (int32_t)i;
        }
    }

    unsigned pending = __ballot_sync(0xffffffffu, count > 32);
    while (pending) {
        const int leader = __ffs(pending) - 1;
        const int32_t lbase = __shfl_sync(0xffffffffu, base, leader);
        const int32_t lcount = __shfl_sync(0xffffffffu, count, leader);
        const int32_t lfirst = __shfl_sync(0xffffffffu, first, leader);
        const int32_t lrow = __shfl_sync(0xffffffffu, (int32_t)i, leader);
        for (int32_t j = lane; j < lcount; j += 32) {
            out_build[lbase + j] = lfirst + j;
            out_probe[lbase + j] = lrow;
        }
        pending &= pending - 1;
    }
}
