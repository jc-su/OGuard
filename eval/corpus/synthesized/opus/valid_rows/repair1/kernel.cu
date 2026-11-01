// valid_rows: the ids of the valid rows of an Arrow validity bitmap.
//
// Each block owns a tile of kTile = 2048 consecutive rows; each warp owns 256
// of them, which are 32 bytes of the bitmap: lane l reads byte l once. In step
// j (0 to 7) the warp takes bit j of every lane's byte, the rows 8 * l + j,
// and a ballot turns them into a 32-bit mask, so a lane's output slot is the
// popcount of the mask bits below it. The warp counts go through a
// block-level exclusive scan, and one atomic add on `count` per block reserves
// the block's output range; warp 0 does the scan and the atomic with warp
// shuffles. The stores of a step go to consecutive slots, so they coalesce.
// The order of the ids is not sorted (the task allows any order).
//
// Bounds: a thread reads its bitmap byte only when the byte is below the
// bitmap length (n + 7) >> 3 and its first row is below n (the two tests are
// equivalent; each states the bound in one form). A store slot comes from the
// atomic on `count`, so it is compared with n before the store; with `count`
// zero on entry every slot is below n, and that test never skips a row. The
// ballots are taken again in the store pass instead of being kept in a
// per-thread array, and the shared array of warp offsets has one entry per
// lane, so any lane index is inside it.
//
// The loads are plain loads, and the warp vote, shuffles, popcount and atomic
// use the NVVM builtins that the CUDA headers wrap: at low optimization levels
// __ldg, __ballot_sync, __shfl_sync, __popc and atomicAdd stay function calls,
// while the builtins compile to single instructions.
//
// Launch: 256 threads per block, ceil(n / 2048) blocks.
#include <cstdint>

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
constexpr int kTile = kBlock * 8;  // one bitmap byte per thread
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// The lanes of the warp whose bit j of `bits` is set.
__device__ __forceinline__ unsigned ballot_bit(uint32_t bits, int j) {
  return __nvvm_vote_ballot_sync(kFullMask, (bits >> j) & 1u);
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
  const int warp = tid >> 5;
  const int64_t nbytes = (n + 7) >> 3;  // the length of the bitmap
  const int64_t byte = static_cast<int64_t>(blockIdx.x) * kBlock + tid;  // this lane's byte
  const int64_t row0 = byte * 8;  // the row of its bit 0

  uint32_t bits = 0;
  if (byte < nbytes && row0 < n) {
    bits = valid[byte];
    if (row0 + 8 > n) bits &= (1u << (n - row0)) - 1u;  // the padding bits past row n
  }
  int total = 0;
  for (int j = 0; j < 8; ++j) total += __nvvm_popc_i(ballot_bit(bits, j));
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
  for (int j = 0; j < 8; ++j) {
    const unsigned mask = ballot_bit(bits, j);
    if ((bits >> j) & 1u) {
      const int64_t slot = pos + __nvvm_popc_i(mask & below);
      if (slot >= 0 && slot < n) rows[slot] = static_cast<int32_t>(row0 + j);
    }
    pos += __nvvm_popc_i(mask);
  }
}
