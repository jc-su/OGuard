// bucketize: out[i] is the number of bounds that are <= values[i]. bounds is
// sorted ascending, so the count is the upper bound of values[i] in bounds,
// which each thread finds by binary search.
#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  const double* __restrict__ bounds,
                                  int32_t* __restrict__ out,
                                  int64_t n,
                                  int64_t b)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const double v = values[i];
        int64_t lo = 0;
        int64_t hi = b;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (bounds[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = static_cast<int32_t>(lo);
    }
}
