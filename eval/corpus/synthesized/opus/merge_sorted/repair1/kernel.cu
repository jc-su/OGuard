// merge_sorted: merge two ascending int64 arrays, a before b on ties.
//
// A merge-path merge. Each block produces the tile out[k0:k0 + tile_len] of at
// most kTile outputs:
//
// 1. Split: find how many of the first k0 (and k1 = k0 + tile_len) outputs
//    come from a, by a binary search along the cross diagonal (lane 0 of warp
//    0 for k0, lane 0 of warp 1 for k1, in parallel). With ties the split
//    takes a first (a[i] <= b[k - 1 - i]). The probes of a binary search close
//    in on the split, near data the block reads next; a 256-ary search with
//    one probe per thread measured 3x slower, as its first rounds read
//    hundreds of scattered sectors per block.
// 2. Load the tile's slices of a and b into shared memory with coalesced reads.
// 3. Each thread finds the split of its own diagonal (kItems outputs apart) in
//    shared memory and merges kItems outputs sequentially into registers.
// 4. The merged tile goes back through shared memory so the stores coalesce.
//
// kItems is odd so that the strided shared-memory accesses of step 4 have no
// bank conflicts.
//
// Bounds: every index is tested or clamped where it is used, against the
// length of its own array, so no bound depends on a loop invariant or on the
// splits that step 1 leaves in shared memory. A search probe into a or b is
// clamped to [0, m - 1] or [0, p - 1] (the search runs only when m > 0 and
// p > 0, and it finds nothing to search otherwise); a tile load tests its
// index against m or p; a shared-memory index is clamped to [0, kTile - 1];
// an output position is compared with m + p. None of these tests changes a
// result for input that meets the contract. The loads are plain loads: at
// low optimization levels __ldg stays a function call.
//
// Launch: 256 threads per block, ceil((m + p) / 1792) blocks.
#include <cstdint>

namespace merge_sorted {
constexpr int kBlock = 256;
constexpr int kItems = 7;
constexpr int kTile = kBlock * kItems;

__device__ __forceinline__ int64_t min64(int64_t x, int64_t y) { return x < y ? x : y; }
__device__ __forceinline__ int64_t max64(int64_t x, int64_t y) { return x > y ? x : y; }
// x clamped to [0, hi].
__device__ __forceinline__ int64_t clamp64(int64_t x, int64_t hi) { return min64(max64(x, 0), hi); }
// A shared-memory slot of the tile: x clamped to [0, kTile - 1].
__device__ __forceinline__ int slot(int x) { return x < 0 ? 0 : (x > kTile - 1 ? kTile - 1 : x); }

// The split of diagonal k: the first i in [max(0, k - p), min(k, m)) with
// a[i] > b[k - 1 - i], or min(k, m) if there is none.
__device__ __forceinline__ int64_t split(const int64_t* __restrict__ a,
                                         const int64_t* __restrict__ b, int64_t m, int64_t p,
                                         int64_t k) {
  int64_t lo = max64(k - p, 0);
  int64_t hi = min64(k, m);
  if (m > 0 && p > 0) {
    while (lo < hi) {
      const int64_t mid = (lo + hi) >> 1;
      if (a[clamp64(mid, m - 1)] <= b[clamp64(k - 1 - mid, p - 1)]) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
  }
  return lo;
}
}  // namespace merge_sorted

extern "C" __global__ void __launch_bounds__(merge_sorted::kBlock)
kernel(const int64_t* __restrict__ a, const int64_t* __restrict__ b,
       int64_t* __restrict__ out, int64_t m, int64_t p) {
  using namespace merge_sorted;
  __shared__ int64_t buf[kTile];
  __shared__ int64_t splits[2];

  const int tid = threadIdx.x;
  const int64_t total = m + p;
  const int64_t k0 = static_cast<int64_t>(blockIdx.x) * kTile;
  // Outputs of this tile (none past the end).
  const int tile_len = static_cast<int>(min64(total - k0, kTile));

  // 1. Split of diagonals k0 and k0 + tile_len.
  if (tid == 0) splits[0] = split(a, b, m, p, k0);
  if (tid == 32) splits[1] = split(a, b, m, p, k0 + tile_len);
  __syncthreads();
  const int64_t a_start = splits[0];
  const int64_t b_start = k0 - a_start;
  const int na = static_cast<int>(min64(max64(splits[1] - a_start, 0), max64(tile_len, 0)));
  const int nb = tile_len - na;

  // 2. Tile inputs: a slice in buf[0:na], b slice in buf[na:tile_len].
  for (int x = tid; x < tile_len; x += kBlock) {
    int64_t v = 0;
    if (x < na) {
      const int64_t i = a_start + x;
      if (i >= 0 && i < m) v = a[i];
    } else {
      const int64_t i = b_start + (x - na);
      if (i >= 0 && i < p) v = b[i];
    }
    buf[x] = v;
  }
  __syncthreads();

  // 3. Split of this thread's diagonal, then a sequential merge of kItems outputs.
  const int d = tid * kItems < tile_len ? tid * kItems : tile_len;
  int lo = d - nb > 0 ? d - nb : 0;
  int hi = d < na ? d : na;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (buf[slot(mid)] <= buf[slot(na + d - 1 - mid)]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int ai = lo;
  int bi = d - lo;
  int64_t vals[kItems];
  for (int j = 0; j < kItems; ++j) {
    const int64_t va = buf[slot(ai)];
    const int64_t vb = buf[slot(na + bi)];
    if (ai < na && (bi >= nb || va <= vb)) {
      vals[j] = va;
      ++ai;
    } else if (bi < nb) {
      vals[j] = vb;
      ++bi;
    }
  }
  __syncthreads();

  // 4. Transpose through shared memory and store coalesced.
  for (int j = 0; j < kItems; ++j) {
    const int x = tid * kItems + j;
    if (x < tile_len) buf[x] = vals[j];
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    const int64_t k = k0 + x;
    if (k < total) out[k] = buf[x];
  }
}
