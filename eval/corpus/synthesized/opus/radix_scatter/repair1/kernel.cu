// radix_scatter: the scatter pass of an LSD radix sort on one byte.
//
// Each block owns a tile of kBlock * kItems consecutive rows and makes one
// atomic add per nonzero digit on the global cursors, instead of one per row:
//
// 1. Rank: each row gets its rank among the tile's rows with the same digit.
//    A warp groups its lanes by digit with a match-any vote, and the lowest
//    lane of each group adds the group size to the block's shared-memory
//    histogram, so a warp makes one shared atomic per distinct digit.
// 2. Reserve: thread d scans the histogram (the tile's local start of digit
//    d) and reserves the digit's output range with one atomic add on
//    cursors[d].
// 3. Scatter: the rows are first sorted by digit in shared memory, then
//    written out in that order, so the stores of a warp go to a few
//    contiguous runs.
//
// The order within a digit is not stable (the task allows any order).
//
// Bounds: every index is tested or clamped where it is used. A key index is
// tested against n; a digit is in [0, 255] because it is masked with 255, and
// the value kRadix that marks a row past the end is tested before it
// indexes; a position in the sorted tile, which comes from shared memory, is
// clamped to [0, kTile - 1]; an output position, which comes from the atomics
// on `cursors`, is compared with n before the store. None of these tests
// changes a result for input that meets the contract.
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
}  // namespace radix_scatter

extern "C" __global__ void __launch_bounds__(radix_scatter::kBlock)
kernel(const uint32_t* __restrict__ keys, int32_t* __restrict__ cursors,
       int32_t* __restrict__ out, int64_t n, int64_t shift) {
  using namespace radix_scatter;
  __shared__ int32_t local_start[kRadix];
  __shared__ int32_t out_offset[kRadix];
  __shared__ int32_t warp_total[kWarps];
  __shared__ int32_t stage_row[kTile];
  __shared__ uint8_t stage_digit[kTile];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t tile_start = static_cast<int64_t>(blockIdx.x) * kTile;
  // Rows of the tile (none past the end).
  const int tile_len = static_cast<int>(n - tile_start < kTile ? n - tile_start : kTile);
  const unsigned s = static_cast<unsigned>(shift);

  // 1. Rank within the tile. local_start holds the histogram here.
  local_start[tid] = 0;
  __syncthreads();
  const uint32_t below = (1u << lane) - 1u;
  int digits[kItems];
  int ranks[kItems];
  for (int k = 0; k < kItems; ++k) {
    const int64_t i = tile_start + k * kBlock + tid;
    // Rows past the end form their own group and are skipped.
    const int digit = i < n ? static_cast<int>((keys[i] >> s) & 255u) : kRadix;
    const uint32_t peers = __match32_any_sync(kFullMask, digit);
    const int leader = lowest_bit(peers);
    int rank = 0;
    if (lane == leader && digit < kRadix) {
      rank = __nvvm_atom_add_gen_i(&local_start[digit & 255], __nvvm_popc_i(peers));
    }
    digits[k] = digit;
    ranks[k] = static_cast<int>(__nvvm_shfl_idx_sync(kFullMask, rank, leader, 0x1f)) +
               __nvvm_popc_i(peers & below);
  }
  __syncthreads();

  // 2. Exclusive scan of the histogram and one global reservation per digit.
  const int c = local_start[tid];
  int inclusive = c;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int t = static_cast<int>(__nvvm_shfl_up_sync(kFullMask, inclusive, offset, 0));
    if (lane >= offset) inclusive += t;
  }
  if (lane == 31) warp_total[warp] = inclusive;
  __syncthreads();
  int start = inclusive - c;
  for (int w = 0; w < warp; ++w) start += warp_total[w];
  const int reserved = c > 0 ? __nvvm_atom_add_gen_i(cursors + tid, c) : 0;
  out_offset[tid] = reserved - start;
  local_start[tid] = start;
  __syncthreads();

  // 3. Sort the tile by digit in shared memory, then write it out in order.
  for (int k = 0; k < kItems; ++k) {
    const int digit = digits[k];
    if (digit < kRadix) {
      int pos = local_start[digit & 255] + ranks[k];
      pos = pos < 0 ? 0 : (pos > kTile - 1 ? kTile - 1 : pos);
      stage_row[pos] = static_cast<int32_t>(tile_start + k * kBlock + tid);
      stage_digit[pos] = static_cast<uint8_t>(digit);
    }
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    const int64_t o = static_cast<int64_t>(out_offset[stage_digit[x]]) + x;
    if (o >= 0 && o < n) out[o] = stage_row[x];
  }
}
