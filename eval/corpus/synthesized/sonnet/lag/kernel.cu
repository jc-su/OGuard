#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ values, const int32_t* __restrict__ first,
                                  double* __restrict__ out, int64_t n, int64_t k) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int64_t j = i - k;
    out[i] = j >= first[i] ? values[j] : __longlong_as_double(0x7FF8000000000000LL);
  }
}
