#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ values, const double* __restrict__ bounds,
                                  int32_t* __restrict__ out, int64_t n, int64_t b) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double v = values[i];
    int64_t lo = 0, hi = b;
    while (lo < hi) {
      int64_t mid = (lo + hi) >> 1;
      if (bounds[mid] <= v) lo = mid + 1;
      else hi = mid;
    }
    out[i] = (int32_t)lo;
  }
}
