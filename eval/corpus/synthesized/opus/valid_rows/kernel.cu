// valid_rows: the ids of the valid rows of an Arrow validity bitmap.
//
// Each block owns a tile of kTile = 2048 consecutive rows; each warp owns 256
// of them, which are 32 bytes of the bitmap: lane l reads byte l once. In step
// j (0 to 7) the warp takes bit j of every lane's byte, the rows 8 * l + j,
// and a ballot turns them into a 32-bit mask, so a lane's output slot is the
// popcount of the mask bits below it. The warp counts go through a
// block-level exclusive scan, and one atomicAdd on `count` per block reserves
// the block's output range. The stores of a step go to consecutive slots, so
// they coalesce. The order of the ids is not sorted (the task allows any
// order).
//
// Launch: 256 threads per block, ceil(n / 2048) blocks.
#include <cstdint>

namespace valid_rows {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kTile = kBlock * 8;  // one bitmap byte per thread
}  // namespace valid_rows

extern "C" __global__ void __launch_bounds__(valid_rows::kBlock)
kernel(const uint8_t* __restrict__ valid, int32_t* __restrict__ rows,
       int32_t* __restrict__ count, int64_t n) {
  using namespace valid_rows;
  __shared__ int32_t warp_offset[kWarps];
  __shared__ int32_t block_base;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // The row of bit 0 of this lane's byte.
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x * 8;

  uint32_t bits = 0;
  if (row0 < n) {
    bits = __ldg(valid + (row0 >> 3));
    if (row0 + 8 > n) bits &= (1u << (n - row0)) - 1u;  // the padding bits past row n
  }
  uint32_t masks[8];
  int total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    masks[j] = __ballot_sync(0xFFFFFFFFu, (bits >> j) & 1u);
    total += __popc(masks[j]);
  }
  if (lane == 0) warp_offset[warp] = total;
  __syncthreads();

  if (threadIdx.x == 0) {
    int running = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int c = warp_offset[w];
      warp_offset[w] = running;
      running += c;
    }
    block_base = running > 0 ? atomicAdd(count, running) : 0;
  }
  __syncthreads();

  int pos = block_base + warp_offset[warp];
  const uint32_t below = (1u << lane) - 1u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if ((bits >> j) & 1u) rows[pos + __popc(masks[j] & below)] = static_cast<int32_t>(row0 + j);
    pos += __popc(masks[j]);
  }
}
