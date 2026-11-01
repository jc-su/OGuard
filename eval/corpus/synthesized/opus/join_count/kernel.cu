// Join count: for each probe key, count the equal keys in the sorted build column.
#include <cstdint>

extern "C" __global__ void kernel(const int64_t* build, const int64_t* probe, int32_t* counts,
                                  int64_t b, int64_t p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < p;
         i += stride) {
        const int64_t key = probe[i];
        // Lower bound: first position whose key is not less than key.
        int64_t lo = 0;
        int64_t hi = b;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (build[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const int64_t first = lo;
        // Upper bound: first position whose key is greater than key.
        hi = b;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (build[mid] <= key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        counts[i] = static_cast<int32_t>(lo - first);
    }
}
