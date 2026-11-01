// min_max: the minimum and maximum of a float64 column.
//
// A single-pass reduction. The launch uses a persistent grid (as many blocks
// as can be resident at once, never more than one per 256 rows); each thread
// folds its grid-stride share of the rows into a running min and max, the
// block combines them with warp shuffles and shared memory, and thread 0 of
// each block applies them to result with one atomic min and one atomic max.
//
// CUDA has no floating-point atomic min/max, so they are built from integer
// atomics on the bit pattern: for a value with the sign bit clear, the order
// of the doubles is the order of their bits as signed integers; for a value
// with the sign bit set, it is the reverse of the order of their bits as
// unsigned integers. Either way the value compares correctly against a stored
// value of either sign. The values contain no NaN.
//
// Launch: 256 threads per block,
// min(ceil(n / 256), SMs * resident blocks per SM) blocks.
#include <cstdint>

namespace min_max {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;

__device__ __forceinline__ void atomic_min(double* address, double value) {
  const long long bits = __double_as_longlong(value);
  if (!signbit(value)) {
    atomicMin(reinterpret_cast<long long*>(address), bits);
  } else {
    atomicMax(reinterpret_cast<unsigned long long*>(address),
              static_cast<unsigned long long>(bits));
  }
}

__device__ __forceinline__ void atomic_max(double* address, double value) {
  const long long bits = __double_as_longlong(value);
  if (!signbit(value)) {
    atomicMax(reinterpret_cast<long long*>(address), bits);
  } else {
    atomicMin(reinterpret_cast<unsigned long long*>(address),
              static_cast<unsigned long long>(bits));
  }
}
}  // namespace min_max

extern "C" __global__ void __launch_bounds__(min_max::kBlock)
kernel(const double* __restrict__ values, double* __restrict__ result, int64_t n) {
  using namespace min_max;
  __shared__ double warp_min[kWarps];
  __shared__ double warp_max[kWarps];

  double lo = __longlong_as_double(0x7FF0000000000000LL);   // +inf
  double hi = __longlong_as_double(static_cast<long long>(0xFFF0000000000000ULL));  // -inf
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kBlock;
#pragma unroll 4
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x; i < n;
       i += stride) {
    const double v = __ldg(values + i);
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lo = fmin(lo, __shfl_down_sync(0xFFFFFFFFu, lo, offset));
    hi = fmax(hi, __shfl_down_sync(0xFFFFFFFFu, hi, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_min[warp] = lo;
    warp_max[warp] = hi;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      lo = fmin(lo, warp_min[w]);
      hi = fmax(hi, warp_max[w]);
    }
    atomic_min(result, lo);
    atomic_max(result + 1, hi);
  }
}
