#include <stdint.h>

// best[k] holds the row of the current maximum of group k, so the input array supplies
// its value and no other scratch is needed. A row replaces the stored row by
// compare-and-swap while it beats it: a larger value, or an equal value at a smaller row.
extern "C" __global__ void kernel(const int32_t* __restrict__ group,
                                  const int64_t* __restrict__ value, int32_t* best,
                                  int64_t n, int64_t g) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;

    const int32_t row = (int32_t)i;
    const int64_t v = value[i];
    int32_t* slot = best + group[i];
    // A stale read is safe: stored rows only improve, so a stale row loses to the candidate
    // only if the current one does, and a failed swap returns the current row.
    int32_t cur = *(volatile int32_t*)slot;
    while (true) {
        if (cur >= 0) {
            int64_t cv = value[cur];
            if (cv > v || (cv == v && cur < row)) break;
        }
        int32_t old = atomicCAS(slot, cur, row);
        if (old == cur) break;
        cur = old;
    }
}
