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
// Bounds: every access is proved by the code at the access alone, without a
// minimum, a maximum or a loop invariant. A global access runs only under the
// two comparisons 0 <= index and index < length of the very index it uses (r
// for run_ends and run_values, n for out). The shared arrays have kSlots =
// 2048 >= kTile entries and every index into them is masked with kSlots - 1.
// A loop over the tile advances its slot with that mask
// (slot = (slot + step) & mask), so that the compiler forms each address from
// the masked or tested index instead of a pointer that steps with the loop.
// None of these tests changes a result for input that meets the contract.
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
constexpr int kSlots = 2048;  // a power of two >= kTile: the size of a shared-memory tile
constexpr int kMask = kSlots - 1;
constexpr int32_t kInt32Max = 2147483647;

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
__device__ __forceinline__ int min32(int a, int b) { return a < b ? a : b; }
}  // namespace run_end_decode

extern "C" __global__ void __launch_bounds__(run_end_decode::kBlock)
kernel(const double* __restrict__ run_values, const int32_t* __restrict__ run_ends,
       double* __restrict__ out, int64_t r, int64_t n) {
  using namespace run_end_decode;
  __shared__ int32_t ends[kSlots];
  __shared__ double stage[kSlots];

  const int tid = threadIdx.x;
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTile;
  // Rows of the tile (none past the end of the column).
  const int tile_len = static_cast<int>(min64(n - t0, kTile));
  const int64_t t1 = t0 + tile_len;

  // 1. First j in [0, r - 1] with run_ends[j] > t0; run_ends[r - 1] = n > t0.
  //    Invariant: that j lies in [below + 1, hi], so run_ends[below] <= t0.
  int64_t first_run = 0;
  if (r > 1) {
    int64_t below = -1;
    int64_t hi = r - 1;
    while (below + 1 < hi) {
      const int64_t step = (hi - below + kBlock - 2) / kBlock;  // ceil((hi - below - 1) / kBlock)
      const int64_t probe = below + (tid + 1) * step;
      bool at_or_before = false;
      if (probe < hi && probe >= 0 && probe < r) at_or_before = run_ends[probe] <= t0;
      below += __nvvm_bar0_popc(at_or_before) * step;
      hi = min64(below + step, hi);
    }
    first_run = below + 1;
  }

  // 2. Stage run ends until a chunk holds the run that reaches t1.
  int loaded = 0;
  int slot = tid & kMask;  // loaded + tid
  while (loaded + kBlock <= kTile) {
    const int64_t j = first_run + slot;
    int32_t e = kInt32Max;
    if (j >= 0 && j < r) e = run_ends[j];
    ends[slot] = e;
    loaded += kBlock;
    slot = (slot + kBlock) & kMask;
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
        if (ends[mid & kMask] <= row0) {
          a = mid + 1;
        } else {
          b = mid;
        }
      }
    }
    int x = x0 & kMask;  // x0 + j, advanced with the mask
    for (int j = 0; j < kItems; ++j) {
      if (x0 + j < tile_len) {
        if (ends[a & kMask] <= row0 + j) ++a;
        const int64_t v = first_run + a;
        if (v >= 0 && v < r) stage[x] = run_values[v];
      }
      x = (x + 1) & kMask;
    }
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    const int s = x & kMask;  // x itself
    const int64_t g = t0 + s;
    if (g >= 0 && g < n) out[g] = stage[s];
  }
}
