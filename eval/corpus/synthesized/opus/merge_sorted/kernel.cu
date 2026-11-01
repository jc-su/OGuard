// merge_sorted: merge two ascending int64 arrays, a before b on ties.
//
// A merge-path merge. Each block produces the tile out[k0:k1] of kTile outputs:
//
// 1. Split: find how many of the first k0 (and k1) outputs come from a, by a
//    binary search along the cross diagonal (lane 0 of warp 0 for k0, lane 0
//    of warp 1 for k1, in parallel). With ties the split takes a first
//    (a[i] <= b[k - 1 - i]). The probes of a binary search close in on the
//    split, near data the block reads next; a 256-ary search with one probe
//    per thread measured 3x slower, as its first rounds read hundreds of
//    scattered sectors per block.
// 2. Load the tile's slices of a and b into shared memory with coalesced reads.
// 3. Each thread finds the split of its own diagonal (kItems outputs apart) in
//    shared memory and merges kItems outputs sequentially into registers.
// 4. The merged tile goes back through shared memory so the stores coalesce.
//
// kItems is odd so that the strided shared-memory accesses of step 4 have no
// bank conflicts.
//
// Launch: 256 threads per block, ceil((m + p) / 1792) blocks.
#include <cstdint>

namespace merge_sorted {
constexpr int kBlock = 256;
constexpr int kItems = 7;
constexpr int kTile = kBlock * kItems;

// The split of diagonal k: the first i in [max(0, k - p), min(k, m)) with
// a[i] > b[k - 1 - i], or min(k, m) if there is none.
__device__ __forceinline__ int64_t split(const int64_t* __restrict__ a,
                                         const int64_t* __restrict__ b, int64_t m, int64_t p,
                                         int64_t k) {
  int64_t lo = k - p > 0 ? k - p : 0;
  int64_t hi = k < m ? k : m;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (__ldg(a + mid) <= __ldg(b + (k - 1 - mid))) {
      lo = mid + 1;
    } else {
      hi = mid;
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

  const int tid = threadIdx.x;
  const int64_t k0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int64_t k1 = k0 + kTile < m + p ? k0 + kTile : m + p;

  // 1. Split of diagonals k0 and k1.
  __shared__ int64_t splits[2];
  if (tid == 0) splits[0] = split(a, b, m, p, k0);
  if (tid == 32) splits[1] = split(a, b, m, p, k1);
  __syncthreads();
  const int64_t lo0 = splits[0];
  const int64_t lo1 = splits[1];
  const int64_t a_start = lo0;
  const int64_t b_start = k0 - lo0;
  const int na = static_cast<int>(lo1 - lo0);
  const int nb = static_cast<int>((k1 - lo1) - b_start);
  const int tile_len = na + nb;

  // 2. Tile inputs: a slice in buf[0:na], b slice in buf[na:tile_len].
  for (int x = tid; x < tile_len; x += kBlock) {
    buf[x] = x < na ? __ldg(a + a_start + x) : __ldg(b + b_start + (x - na));
  }
  __syncthreads();

  // 3. Split of this thread's diagonal, then a sequential merge of kItems outputs.
  const int d = tid * kItems < tile_len ? tid * kItems : tile_len;
  int lo = d - nb > 0 ? d - nb : 0;
  int hi = d < na ? d : na;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (buf[mid] <= buf[na + d - 1 - mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int ai = lo;
  int bi = d - lo;
  int64_t vals[kItems];
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    if (ai < na && (bi >= nb || buf[ai] <= buf[na + bi])) {
      vals[j] = buf[ai++];
    } else if (bi < nb) {
      vals[j] = buf[na + bi++];
    }
  }
  __syncthreads();

  // 4. Transpose through shared memory and store coalesced.
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int x = tid * kItems + j;
    if (x < tile_len) buf[x] = vals[j];
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) out[k0 + x] = buf[x];
}
