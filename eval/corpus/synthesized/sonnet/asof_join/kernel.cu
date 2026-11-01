#include <cstdint>

extern "C" __global__ void kernel(const int64_t* left, const int64_t* right,
                                  int32_t* out, int64_t n, int64_t r,
                                  int64_t tolerance) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    int64_t x = left[i];
    // Upper bound: the first row of right that is greater than x.
    int64_t lo = 0, hi = r;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (right[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    // The largest row with right[j] <= x is the closest to x, so if it is
    // outside the tolerance every smaller row is too.
    int64_t j = lo - 1;
    out[i] = (j >= 0 && x - right[j] <= tolerance) ? (int32_t)j : -1;
}
