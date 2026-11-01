// Task 12, segmented_sum: sums[j] = sum of values[offsets[j], offsets[j + 1]).
//
// The kernel selects one of three mappings from the average segment length
// (offsets[g] - offsets[0]) / g. All threads compute the same average, so the
// selection is uniform over the grid:
//   - average >= kBlockModeMinAverage: one block per segment.
//   - average >= kWarpModeMinAverage: one warp per segment.
//   - otherwise: one thread per segment. A segment of kWarpSize or more
//     values is summed by the whole warp of its thread instead, so that one
//     long segment does not stall a warp of short ones.
// Each mapping uses a grid-stride loop, so every grid size gives the correct
// result; the grid size only changes the speed. An empty segment gives 0.0.
// Each segment is summed in a fixed order, so the result is deterministic.
//
// Launch requirement: blockDim.x is a multiple of 32, so that each group of 32
// consecutive threads is one hardware warp.
#include <cstdint>

namespace segmented_sum {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr int64_t kWarpModeMinAverage = 32;
constexpr int64_t kBlockModeMinAverage = 1024;

// Returns the sum of x over the warp in lane 0.
__device__ __forceinline__ double warp_sum(double x) {
  for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
    x += __shfl_down_sync(kFullMask, x, delta);
  }
  return x;
}

// Returns the sum of values[i], values[i + step], ... below end. Four
// independent accumulators keep four loads in flight per thread.
__device__ __forceinline__ double strided_sum(const double* __restrict__ values, int64_t i,
                                              int64_t end, int64_t step) {
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  for (; i + 3 * step < end; i += 4 * step) {
    a0 += values[i];
    a1 += values[i + step];
    a2 += values[i + 2 * step];
    a3 += values[i + 3 * step];
  }
  for (; i < end; i += step) a0 += values[i];
  return (a0 + a1) + (a2 + a3);
}

}  // namespace segmented_sum

extern "C" __global__ void kernel(const int32_t* __restrict__ offsets,
                                  const double* __restrict__ values,
                                  double* __restrict__ sums,
                                  int64_t g) {
  using namespace segmented_sum;
  const int64_t total = static_cast<int64_t>(offsets[g]) - offsets[0];
  const int lane = threadIdx.x % kWarpSize;

  if (total >= kBlockModeMinAverage * g) {
    // One block per segment: each thread sums a strided part, then the warp
    // sums go through shared memory to warp 0.
    __shared__ double warp_sums[kWarpSize];
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    for (int64_t j = blockIdx.x; j < g; j += gridDim.x) {
      double acc = warp_sum(strided_sum(values, offsets[j] + threadIdx.x, offsets[j + 1],
                                        blockDim.x));
      if (lane == 0) warp_sums[warp] = acc;
      __syncthreads();
      if (warp == 0) {
        acc = warp_sum(lane < warps ? warp_sums[lane] : 0.0);
        if (lane == 0) sums[j] = acc;
      }
      // warp_sums is written again in the next iteration.
      __syncthreads();
    }
    return;
  }

  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int64_t warps = (static_cast<int64_t>(gridDim.x) * blockDim.x) / kWarpSize;

  if (total >= kWarpModeMinAverage * g) {
    // One warp per segment. j is the same in all lanes, so all lanes reach warp_sum.
    for (int64_t j = warp; j < g; j += warps) {
      const int64_t end = offsets[j + 1];
      double acc = 0.0;
      for (int64_t i = offsets[j] + lane; i < end; i += kWarpSize) acc += values[i];
      acc = warp_sum(acc);
      if (lane == 0) sums[j] = acc;
    }
    return;
  }

  // One thread per segment. The loop runs over groups of 32 segments, one
  // group per warp, so base is the same in all lanes.
  for (int64_t base = warp * kWarpSize; base < g; base += warps * kWarpSize) {
    const int64_t j = base + lane;
    int64_t begin = 0;
    int64_t end = 0;
    if (j < g) {
      begin = offsets[j];
      end = offsets[j + 1];
    }
    const bool is_long = end - begin >= kWarpSize;
    // The whole warp sums each long segment of the group, one at a time.
    unsigned pending = __ballot_sync(kFullMask, is_long);
    while (pending != 0) {
      const int src = __ffs(pending) - 1;
      pending &= pending - 1;
      const int64_t long_end = __shfl_sync(kFullMask, end, src);
      double acc = 0.0;
      for (int64_t i = __shfl_sync(kFullMask, begin, src) + lane; i < long_end; i += kWarpSize) {
        acc += values[i];
      }
      acc = warp_sum(acc);
      if (lane == 0) sums[base + src] = acc;
    }
    if (j < g && !is_long) {
      double acc = 0.0;
      for (int64_t i = begin; i < end; ++i) acc += values[i];
      sums[j] = acc;
    }
  }
}
