// radix_scatter: the scatter pass of an LSD radix sort on one byte.
//
// Each block owns a tile of kBlock * kItems consecutive rows and makes one
// atomicAdd per nonzero digit on the global cursors, instead of one per row:
//
// 1. Rank: each row gets its rank among the tile's rows with the same digit.
//    A warp groups its lanes by digit with __match_any_sync, and the lowest
//    lane of each group adds the group size to the block's shared-memory
//    histogram, so a warp makes one shared atomic per distinct digit.
// 2. Reserve: thread d scans the histogram (the tile's local start of digit
//    d) and reserves the digit's output range with one atomicAdd on
//    cursors[d].
// 3. Scatter: the rows are first sorted by digit in shared memory, then
//    written out in that order, so the stores of a warp go to a few
//    contiguous runs.
//
// The order within a digit is not stable (the task allows any order).
//
// Launch: 256 threads per block, ceil(n / 2048) blocks.
#include <cstdint>

namespace radix_scatter {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kItems = 8;
constexpr int kTile = kBlock * kItems;
constexpr int kRadix = 256;
static_assert(kRadix == kBlock, "the reserve step gives one digit to each thread");
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
  const int tile_len = static_cast<int>(n - tile_start < kTile ? n - tile_start : kTile);
  const unsigned s = static_cast<unsigned>(shift);

  // 1. Rank within the tile. local_start holds the histogram here.
  local_start[tid] = 0;
  __syncthreads();
  const uint32_t below = (1u << lane) - 1u;
  int digits[kItems];
  int ranks[kItems];
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int x = k * kBlock + tid;
    // Rows past the end form their own group and are skipped.
    const int digit = x < tile_len ? static_cast<int>((__ldg(keys + tile_start + x) >> s) & 255u)
                                   : kRadix;
    const uint32_t peers = __match_any_sync(0xFFFFFFFFu, digit);
    const int leader = __ffs(peers) - 1;
    int rank = 0;
    if (lane == leader && digit < kRadix) rank = atomicAdd(&local_start[digit], __popc(peers));
    digits[k] = digit;
    ranks[k] = __shfl_sync(0xFFFFFFFFu, rank, leader) + __popc(peers & below);
  }
  __syncthreads();

  // 2. Exclusive scan of the histogram and one global reservation per digit.
  const int c = local_start[tid];
  int inclusive = c;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int t = __shfl_up_sync(0xFFFFFFFFu, inclusive, offset);
    if (lane >= offset) inclusive += t;
  }
  if (lane == 31) warp_total[warp] = inclusive;
  __syncthreads();
  int start = inclusive - c;
  for (int w = 0; w < warp; ++w) start += warp_total[w];
  const int reserved = c > 0 ? atomicAdd(cursors + tid, c) : 0;
  out_offset[tid] = reserved - start;
  local_start[tid] = start;
  __syncthreads();

  // 3. Sort the tile by digit in shared memory, then write it out in order.
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    if (digits[k] < kRadix) {
      const int pos = local_start[digits[k]] + ranks[k];
      stage_row[pos] = static_cast<int32_t>(tile_start + k * kBlock + tid);
      stage_digit[pos] = static_cast<uint8_t>(digits[k]);
    }
  }
  __syncthreads();
  for (int x = tid; x < tile_len; x += kBlock) {
    out[out_offset[stage_digit[x]] + x] = stage_row[x];
  }
}
