// run_end_decode: decode an Arrow run-end encoded float64 column.
//
// Each block decodes a tile of kTile consecutive rows [t0, t0 + tile_len):
//
// 1. Find the first run of the tile, the first j with run_ends[j] > t0, by a
//    256-ary search: in each round every thread tests one splitter of the
//    remaining range and a block-wide count gives the sub-range that holds
//    the answer. A range of r runs takes ceil(log256(r)) rounds of one load
//    per thread, instead of log2(r) dependent loads. The first rounds probe
//    the same positions in every block, so they hit L2.
// 2. Stage the ends of the runs that cover the tile in shared memory, one
//    coalesced chunk of kBlock runs at a time, until a chunk holds the run
//    that reaches the tile end. The tile's kTile rows meet at most kTile runs.
// 3. Each thread decodes kItems consecutive rows: a binary search over the
//    staged ends finds the run of its first row (row t0 + x lies in one of the
//    first x + 1 staged runs), and each next row is in the same run or the
//    next one, which one comparison decides. The values go through shared
//    memory so that the stores coalesce. kItems is odd so that the strided
//    shared-memory accesses have no bank conflicts.
//
// Bounds: every index is tested or clamped where it is used, against the
// length of its own array. A search probe is clamped to [0, r - 1] (it is
// already in that range when probe < hi), a run index is compared with r, an
// output row with n, and a shared-memory index is clamped to kTile - 1 or
// bounded by tile_len = min(n - t0, kTile). None of these tests changes a
// result for input that meets the contract.
//
// The loads are plain loads, and the block-wide count uses the NVVM builtin
// that __syncthreads_count wraps: at low optimization levels __ldg and
// __syncthreads_count stay function calls, while the builtin compiles to one
// bar.red.popc instruction.
//
// Launch: 256 threads per block, ceil(n / 1792) blocks.
#include <cstdint>

extern __device__ __device_builtin__ int __nvvm_bar0_popc(int predicate);

namespace run_end_decode {
constexpr int kBlock = 256;
constexpr int kItems = 7;
constexpr int kTile = kBlock * kItems;
constexpr int32_t kInt32Max = 2147483647;

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }
__device__ __forceinline__ int min32(int a, int b) { return a < b ? a : b; }
}  // namespace run_end_decode

extern "C" __global__ void __launch_bounds__(run_end_decode::kBlock)
kernel(const double* __restrict__ run_values, const int32_t* __restrict__ run_ends,
       double* __restrict__ out, int64_t r, int64_t n) {
  using namespace run_end_decode;
  __shared__ int32_t ends[kTile];
  __shared__ double stage[kTile];

  const int tid = threadIdx.x;
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTile;
  // Rows of the tile (none past the end of the column).
  const int tile_len = static_cast<int>(min64(n - t0, kTile));
  const int64_t t1 = t0 + tile_len;

  // 1. First j in [0, r - 1] with run_ends[j] > t0; run_ends[r - 1] = n > t0.
  int64_t first_run = 0;
  if (r > 1) {
    const int64_t last = r - 1;
    int64_t lo = 0;
    int64_t hi = last;
    while (lo < hi) {
      const int64_t step = (hi - lo + kBlock - 1) / kBlock;
      const int64_t probe = lo + (tid + 1) * step - 1;
      const bool below = probe < hi && run_ends[min64(max64(probe, 0), last)] <= t0;
      lo += __nvvm_bar0_popc(below) * step;
      hi = min64(lo + step - 1, hi);
    }
    first_run = max64(lo, 0);
  }

  // 2. Stage run ends until a chunk holds the run that reaches t1.
  int loaded = 0;
  while (loaded + kBlock <= kTile) {
    const int slot = loaded + tid;
    const int64_t j = first_run + slot;
    const int32_t e = j < r ? run_ends[j] : kInt32Max;
    ends[slot] = e;
    loaded += kBlock;
    if (__nvvm_bar0_popc(e >= t1) > 0) break;
  }

  // 3. Decode rows t0 + [x0, x0 + kItems): the run of row t0 + x is the first
  //    staged a with ends[a] > row.
  const int x0 = tid * kItems;
  if (x0 < tile_len) {
    const int64_t row0 = t0 + x0;
    int a = 0;
    if (ends[0] <= row0) {
      a = 1;
      int b = min32(loaded, x0 + 1);
      while (a < b) {
        const int mid = (a + b) >> 1;
        if (ends[min32(mid, kTile - 1)] <= row0) {
          a = mid + 1;
        } else {
          b = mid;
        }
      }
    }
    for (int j = 0; j < kItems; ++j) {
      const int x = x0 + j;
      if (x < tile_len) {
        if (ends[min32(a, kTile - 1)] <= row0 + j) ++a;
        const int64_t v = first_run + a;
        if (v < r) stage[x] = run_values[v];
      }
    }
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    const int64_t row = t0 + x;
    if (row < n) out[row] = stage[x];
  }
}
