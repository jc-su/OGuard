// radix_scatter: the scatter pass of an LSD radix sort on one byte.
//
// Each block owns a tile of kBlock * kItems consecutive rows and makes one
// atomic add per nonzero digit on the global cursors, instead of one per row:
//
// 1. Rank: each row gets its rank among the tile's rows with the same digit.
//    A warp groups its lanes by digit with a match-any vote, and the lowest
//    lane of each group adds the group size to the block's shared-memory
//    histogram, so a warp makes one shared atomic per distinct digit. The
//    row's digit and rank go to shared memory, packed as digit << 16 | rank.
// 2. Reserve: thread d scans the histogram (the tile's local start of digit
//    d) and reserves the digit's output range with one atomic add on
//    cursors[d].
// 3. Scatter: the rows are first sorted by digit in shared memory, then
//    written out in that order, so the stores of a warp go to a few
//    contiguous runs.
//
// The order within a digit is not stable (the task allows any order).
//
// Bounds: every access is proved by the code at the access alone, without a
// minimum, a maximum or a loop invariant. A key is read and a row id stored
// only under the two comparisons 0 <= index and index < n of the very index
// used; an output position comes from the atomics on `cursors`. An index into
// the 256-entry arrays (cursors, the histogram and the output offsets) is
// masked with 255, the warp index with kWarps - 1, and an index into a
// tile-sized shared array with kTile - 1. A loop over the tile takes its
// position through that mask too, so that the compiler forms each address
// from the tested index instead of a pointer that steps with the loop; a loop
// with a fixed count advances its slot with the mask
// (slot = (slot + kBlock) & mask) for the same reason. None of these changes
// a result for input that meets the contract and a launch with kBlock threads
// per block. The per-row digits and ranks live in shared memory, so the
// kernel keeps no per-thread array in local memory.
//
// The loads are plain loads, and the warp votes, shuffles, bit counts and
// atomics use the NVVM builtins that the CUDA headers wrap: at low
// optimization levels __ldg, __match_any_sync, __shfl_sync, __popc, __ffs and
// atomicAdd stay function calls, while the builtins compile to single
// instructions.
//
// Launch: 256 threads per block, ceil(n / 2048) blocks.
#include <cstdint>

extern __device__ __device_builtin__ unsigned int __match32_any_sync(unsigned int mask,
                                                                     unsigned int value);
extern __device__ __device_builtin__ unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a,
                                                                   unsigned b, unsigned c);
extern __device__ __device_builtin__ unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a,
                                                                  unsigned b, unsigned c);
extern __device__ __device_builtin__ int __nvvm_popc_i(int x);
extern __device__ __device_builtin__ int __nvvm_clz_i(int x);
extern __device__ __device_builtin__ int __nvvm_atom_add_gen_i(volatile int* address, int value);

namespace radix_scatter {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kItems = 8;
constexpr int kTile = kBlock * kItems;
constexpr int kRadix = 256;
constexpr unsigned kFullMask = 0xFFFFFFFFu;
static_assert(kRadix == kBlock, "the reserve step gives one digit to each thread");

// The index of the lowest set bit of a nonzero x.
__device__ __forceinline__ int lowest_bit(unsigned x) { return 31 - __nvvm_clz_i(x & (0u - x)); }
static_assert((kTile & (kTile - 1)) == 0, "a tile slot is masked with kTile - 1");

// A slot of a tile-sized shared array: x masked with kTile - 1.
__device__ __forceinline__ int slot(int x) { return x & (kTile - 1); }
}  // namespace radix_scatter

extern "C" __global__ void __launch_bounds__(radix_scatter::kBlock)
kernel(const uint32_t* __restrict__ keys, int32_t* __restrict__ cursors,
       int32_t* __restrict__ out, int64_t n, int64_t shift) {
  using namespace radix_scatter;
  __shared__ int32_t local_start[kRadix];
  __shared__ int32_t out_offset[kRadix];
  __shared__ int32_t warp_total[kWarps];
  __shared__ int32_t digit_rank[kTile];
  __shared__ int32_t stage_row[kTile];
  __shared__ uint8_t stage_digit[kTile];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bin = tid & (kRadix - 1);  // the digit of this thread in the reserve step
  const int64_t tile_start = static_cast<int64_t>(blockIdx.x) * kTile;
  // Rows of the tile (none past the end).
  const int tile_len = static_cast<int>(n - tile_start < kTile ? n - tile_start : kTile);
  const unsigned s = static_cast<unsigned>(shift);

  // 1. Rank within the tile. local_start holds the histogram here.
  local_start[bin] = 0;
  __syncthreads();
  const uint32_t below = (1u << lane) - 1u;
  int sx = slot(tid);  // the row's place in the tile, k * kBlock + tid
  for (int k = 0; k < kItems; ++k) {
    const int64_t i = tile_start + sx;
    // Rows past the end form their own group and are skipped.
    int digit = kRadix;
    if (i >= 0 && i < n) digit = static_cast<int>((keys[i] >> s) & 255u);
    const uint32_t peers = __match32_any_sync(kFullMask, digit);
    const int leader = lowest_bit(peers);
    int rank = 0;
    if (lane == leader && digit < kRadix) {
      rank = __nvvm_atom_add_gen_i(&local_start[digit & (kRadix - 1)], __nvvm_popc_i(peers));
    }
    rank = static_cast<int>(__nvvm_shfl_idx_sync(kFullMask, rank, leader, 0x1f)) +
           __nvvm_popc_i(peers & below);
    digit_rank[sx] = (digit << 16) | rank;
    sx = slot(sx + kBlock);
  }
  __syncthreads();

  // 2. Exclusive scan of the histogram and one global reservation per digit.
  const int c = local_start[bin];
  int inclusive = c;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int t = static_cast<int>(__nvvm_shfl_up_sync(kFullMask, inclusive, offset, 0));
    if (lane >= offset) inclusive += t;
  }
  if (lane == 31) warp_total[warp & (kWarps - 1)] = inclusive;
  __syncthreads();
  int start = inclusive - c;
  for (int w = 0; w < warp; ++w) start += warp_total[w & (kWarps - 1)];
  const int reserved = c > 0 ? __nvvm_atom_add_gen_i(cursors + bin, c) : 0;
  out_offset[bin] = reserved - start;
  local_start[bin] = start;
  __syncthreads();

  // 3. Sort the tile by digit in shared memory, then write it out in order.
  sx = slot(tid);
  for (int k = 0; k < kItems; ++k) {
    const int v = digit_rank[sx];
    const int digit = v >> 16;
    if (digit < kRadix) {
      const int pos = slot(local_start[digit & (kRadix - 1)] + (v & 0xFFFF));
      stage_row[pos] = static_cast<int32_t>(tile_start + sx);
      stage_digit[pos] = static_cast<uint8_t>(digit);
    }
    sx = slot(sx + kBlock);
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    const int sx = slot(x);
    const int64_t o = static_cast<int64_t>(out_offset[stage_digit[sx]]) + sx;
    if (o >= 0 && o < n) out[o] = stage_row[sx];
  }
}
