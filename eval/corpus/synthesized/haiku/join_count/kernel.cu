#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* build,
    const int64_t* probe,
    int32_t* counts,
    int64_t b,
    int64_t p
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p) {
        int64_t probe_val = probe[idx];
        int32_t count = 0;

        // Binary search in sorted build array
        int64_t lo = 0, hi = b;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (build[mid] < probe_val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Count equal elements
        while (lo < b && build[lo] == probe_val) {
            count++;
            lo++;
        }

        counts[idx] = count;
    }
}
