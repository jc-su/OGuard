// run_end_decode: decode an Arrow run-end encoded float64 column.
//
// Each block decodes a tile of kTile consecutive rows [t0, t1):
//
// 1. Find the first run of the tile, the first j with run_ends[j] > t0, by a
//    256-ary search: in each round every thread tests one splitter of the
//    remaining range and __syncthreads_count gives the sub-range that holds
//    the answer. A range of r runs takes ceil(log256(r)) rounds of one load
//    per thread, instead of log2(r) dependent loads. The first rounds probe
//    the same positions in every block, so they hit L2.
// 2. Stage the ends of the runs that cover the tile in shared memory, one
//    coalesced chunk of kBlock runs at a time, until a chunk holds the run
//    that reaches t1. The tile's kTile rows meet at most kTile runs.
// 3. Each thread decodes kItems consecutive rows: a binary search over the
//    staged ends finds the run of its first row (row t0 + x lies in one of the
//    first x + 1 staged runs), and each next row is in the same run or the
//    next one, which one comparison decides. The values go through shared
//    memory so that the stores coalesce. kItems is odd so that the strided
//    shared-memory accesses have no bank conflicts.
//
// Launch: 256 threads per block, ceil(n / 1792) blocks.
#include <cstdint>

namespace run_end_decode {
constexpr int kBlock = 256;
constexpr int kItems = 7;
constexpr int kTile = kBlock * kItems;
constexpr int32_t kInt32Max = 2147483647;
}  // namespace run_end_decode

extern "C" __global__ void __launch_bounds__(run_end_decode::kBlock)
kernel(const double* __restrict__ run_values, const int32_t* __restrict__ run_ends,
       double* __restrict__ out, int64_t r, int64_t n) {
  using namespace run_end_decode;
  __shared__ int32_t ends[kTile];
  __shared__ double stage[kTile];

  const int tid = threadIdx.x;
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t t1 = t0 + kTile < n ? t0 + kTile : n;

  // 1. First j in [0, r - 1] with run_ends[j] > t0; run_ends[r - 1] = n > t0.
  int64_t lo = 0;
  int64_t hi = r - 1;
  while (lo < hi) {
    const int64_t step = (hi - lo + kBlock - 1) / kBlock;
    const int64_t probe = lo + (tid + 1) * step - 1;
    lo += __syncthreads_count(probe < hi && __ldg(run_ends + probe) <= t0) * step;
    hi = lo + step - 1 < hi ? lo + step - 1 : hi;
  }
  const int64_t first_run = lo;

  // 2. Stage run ends until a chunk holds the run that reaches t1.
  int loaded = 0;
  for (int c = 0; c < kItems; ++c) {
    const int64_t j = first_run + c * kBlock + tid;
    const int32_t e = j < r ? __ldg(run_ends + j) : kInt32Max;
    ends[c * kBlock + tid] = e;
    loaded += kBlock;
    if (__syncthreads_count(e >= t1) > 0) break;
  }

  // 3. Decode rows t0 + [x0, x0 + kItems): the run of row t0 + x is the first
  //    staged a with ends[a] > row.
  const int tile_len = static_cast<int>(t1 - t0);
  const int x0 = tid * kItems;
  if (x0 < tile_len) {
    const int32_t row0 = static_cast<int32_t>(t0 + x0);
    int a = 0;
    if (ends[0] <= row0) {
      a = 1;
      int b = loaded < x0 + 1 ? loaded : x0 + 1;
      while (a < b) {
        const int mid = (a + b) >> 1;
        if (ends[mid] <= row0) {
          a = mid + 1;
        } else {
          b = mid;
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      const int x = x0 + j;
      if (x < tile_len) {
        if (ends[a] <= row0 + j) ++a;
        stage[x] = __ldg(run_values + first_run + a);
      }
    }
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) out[t0 + x] = stage[x];
}
