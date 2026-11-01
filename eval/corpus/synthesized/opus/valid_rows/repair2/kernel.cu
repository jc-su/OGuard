// valid_rows: the ids of the valid rows of an Arrow validity bitmap.
//
// Each block owns a tile of kTile = 2048 consecutive rows; each warp owns 256
// of them. In step j (0 to 7) lane l of a warp takes the row 32 * j + l of the
// warp's range, reads the bitmap byte row >> 3 (the lanes of a step read 4
// consecutive bytes) and its bit row & 7, and a ballot turns the 32 bits into
// a mask, so a lane's output slot is the popcount of the mask bits below it.
// The warp counts go through a block-level exclusive scan, and one atomic add
// on `count` per block reserves the block's output range; warp 0 does the scan
// and the atomic with warp shuffles. The stores of a step go to consecutive
// slots, so they coalesce, and the ids come out ascending within a block. The
// order of the blocks is the order of their atomics (the task allows any
// order).
//
// Bounds: every access is proved by the code at the access alone, without a
// minimum, a maximum or a loop invariant. A thread reads the bitmap byte
// row >> 3 only under 0 <= row and row < n for the row itself (the bitmap has
// (n + 7) / 8 bytes, so row >> 3 is one of them), and under the same test of
// the byte index against (n + 7) >> 3. A store slot comes from the atomic on
// `count` and is stored only under 0 <= slot and slot < n; with `count` zero
// on entry every slot is below n, and that test never skips a row. The
// per-step bits of a thread are kept in the bits of one integer, and the
// shared array of warp counts has one entry per lane, indexed by a lane or
// warp number masked with 31. The reading steps are unrolled in the source
// (a fold over the step numbers), so each step forms its row and byte from
// the thread's first row and a constant, and no byte index steps with a loop.
//
// The loads are plain loads, and the warp vote, shuffles, popcount and atomic
// use the NVVM builtins that the CUDA headers wrap: at low optimization levels
// __ldg, __ballot_sync, __shfl_sync, __popc and atomicAdd stay function calls,
// while the builtins compile to single instructions.
//
// Launch: 256 threads per block, ceil(n / 2048) blocks.
#include <cstdint>
#include <utility>

extern __device__ __device_builtin__ unsigned int __nvvm_vote_ballot_sync(unsigned int mask,
                                                                          int pred);
extern __device__ __device_builtin__ unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a,
                                                                  unsigned b, unsigned c);
extern __device__ __device_builtin__ unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a,
                                                                   unsigned b, unsigned c);
extern __device__ __device_builtin__ int __nvvm_popc_i(int x);
extern __device__ __device_builtin__ int __nvvm_atom_add_gen_i(volatile int* address, int value);

namespace valid_rows {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kSteps = 8;  // rows per thread
constexpr int kTile = kBlock * kSteps;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Step j of the reading pass: reads the bit of row first_row + 32 * j, sets
// bit j of `bits` to it and adds the warp's count of valid rows to `total`.
__device__ __forceinline__ void read_step(int j, const uint8_t* __restrict__ valid, int64_t n,
                                          int64_t nbytes, int64_t first_row, uint32_t& bits,
                                          int& total) {
  const int64_t row = first_row + 32 * j;
  const int64_t byte = row >> 3;
  uint32_t valid_bit = 0;
  if (row >= 0 && row < n && byte >= 0 && byte < nbytes) {
    valid_bit = (static_cast<uint32_t>(valid[byte]) >> (row & 7)) & 1u;
  }
  bits |= valid_bit << j;
  total += __nvvm_popc_i(static_cast<int>(__nvvm_vote_ballot_sync(kFullMask, valid_bit)));
}

// The steps 0, 1, ..., kSteps - 1 of the reading pass, unrolled in the source.
template <int... kStep>
__device__ __forceinline__ void read_steps(const uint8_t* __restrict__ valid, int64_t n,
                                           int64_t nbytes, int64_t first_row, uint32_t& bits,
                                           int& total, std::integer_sequence<int, kStep...>) {
  (read_step(kStep, valid, n, nbytes, first_row, bits, total), ...);
}
}  // namespace valid_rows

extern "C" __global__ void __launch_bounds__(valid_rows::kBlock)
kernel(const uint8_t* __restrict__ valid, int32_t* __restrict__ rows,
       int32_t* __restrict__ count, int64_t n) {
  using namespace valid_rows;
  // Warp counts, then output offsets; one entry per lane of warp 0.
  __shared__ int32_t warp_offset[32];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = (tid >> 5) & 31;
  const int64_t nbytes = (n + 7) >> 3;  // the length of the bitmap
  const int64_t first_row =
      static_cast<int64_t>(blockIdx.x) * kTile + warp * (32 * kSteps) + lane;

  uint32_t bits = 0;  // bit j: the row first_row + 32 * j is valid
  int total = 0;
  read_steps(valid, n, nbytes, first_row, bits, total, std::make_integer_sequence<int, kSteps>{});
  if (lane == 0) warp_offset[warp] = total;
  __syncthreads();

  // Warp 0 scans the warp counts (lane w holds warp w) and reserves the
  // block's output range with one atomic add.
  if (warp == 0) {
    const int c = lane < kWarps ? warp_offset[lane] : 0;
    int inclusive = c;
    for (int offset = 1; offset < kWarps; offset <<= 1) {
      const int t = static_cast<int>(__nvvm_shfl_up_sync(kFullMask, inclusive, offset, 0));
      if (lane >= offset) inclusive += t;
    }
    const int running =
        static_cast<int>(__nvvm_shfl_idx_sync(kFullMask, inclusive, kWarps - 1, 0x1f));
    int base = 0;
    if (lane == 0 && running > 0) base = __nvvm_atom_add_gen_i(count, running);
    base = static_cast<int>(__nvvm_shfl_idx_sync(kFullMask, base, 0, 0x1f));
    if (lane < kWarps) warp_offset[lane] = base + inclusive - c;
  }
  __syncthreads();

  int64_t pos = warp_offset[warp];
  const unsigned below = (1u << lane) - 1u;
  for (int j = 0; j < kSteps; ++j) {
    const uint32_t valid_bit = (bits >> j) & 1u;
    const unsigned mask = __nvvm_vote_ballot_sync(kFullMask, valid_bit);
    if (valid_bit != 0) {
      const int64_t slot = pos + __nvvm_popc_i(static_cast<int>(mask & below));
      if (slot >= 0 && slot < n) rows[slot] = static_cast<int32_t>(first_row + 32 * j);
    }
    pos += __nvvm_popc_i(static_cast<int>(mask));
  }
}
