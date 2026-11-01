// filter_select: write the row ids i with quantity[i] < threshold into sel.
//
// Each block processes one tile of blockDim.x rows per iteration of a
// grid-stride loop, one row per thread. Lane 0 of each warp reserves the range
// of its warp's matches within the tile with an atomicAdd on a shared counter,
// thread 0 reserves the range of the tile in sel with a single atomicAdd on
// count, and each matching thread writes its row id into that range. The
// kernel requires count[0] == 0 on entry, n <= INT32_MAX (sel holds int32 row
// ids), and a block size that is a multiple of 32 and at most BLOCK. The
// caller sets count[0] to zero before the launch (see run_filter_select in
// test.cu).
//
// A thread writes its row id only when its position is in [0, n). If count[0]
// is zero on entry, every position is below the number of matches, which is at
// most n, so the check skips no write. The check keeps each store inside sel
// for any value in count on entry.
//
// The same store also checks the row id itself: i < n and i <= INT32_MAX. A
// matching row has i < n, and the kernel requires n <= INT32_MAX, so the check
// skips no write. The check puts the bound of the stored value next to the
// store, where the value is known to be a row id of quantity that fits in
// int32. The first test of i < n is in the match predicate, which reaches the
// store only through the warp vote.

#include <cstdint>

constexpr int BLOCK = 256;  // Largest number of threads per block.
// Rows per tile of a block of BLOCK threads. The launcher in test.cu uses it.
[[maybe_unused]] constexpr long long TILE = BLOCK;
constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr long long ROW_ID_MAX = INT32_MAX;  // Largest row id that sel can hold.

extern "C" __global__ void __launch_bounds__(BLOCK)
kernel(const double* __restrict__ quantity, double threshold, int32_t* __restrict__ sel,
       int32_t* __restrict__ count, long long n)
{
    // tile_count counts the matches of the current tile. tile_first is the
    // position in sel of the first match of the current tile.
    __shared__ int tile_count;
    __shared__ int tile_first;

    const unsigned lane = threadIdx.x % 32;
    const unsigned lanes_below = (1u << lane) - 1u;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    if (threadIdx.x == 0) {
        tile_count = 0;
    }
    __syncthreads();

    // The loop bounds are the same for all threads of a block, so every thread
    // reaches each __syncthreads call. Thread 0 writes tile_count and
    // tile_first only between the two __syncthreads calls of a tile, when no
    // other thread reads or writes them.
    for (long long tile = static_cast<long long>(blockIdx.x) * blockDim.x; tile < n;
         tile += stride) {
        const long long i = tile + threadIdx.x;
        const bool keep = i < n && quantity[i] < threshold;
        const unsigned mask = __ballot_sync(FULL_MASK, keep);

        // Lane 0 reserves the range of its warp within the tile.
        int warp_first = 0;
        if (lane == 0 && mask != 0) {
            warp_first = atomicAdd(&tile_count, __popc(mask));
        }
        warp_first = __shfl_sync(FULL_MASK, warp_first, 0);
        __syncthreads();

        // Thread 0 reserves the range of the tile in sel.
        if (threadIdx.x == 0) {
            const int matches = tile_count;
            if (matches != 0) {
                tile_first = atomicAdd(count, matches);
            }
            tile_count = 0;
        }
        __syncthreads();

        // Each matching lane writes its row id after the matches of the lower
        // lanes of its warp.
        if (keep) {
            const long long position = static_cast<long long>(tile_first) + warp_first +
                                       __popc(mask & lanes_below);
            if (i < n && i <= ROW_ID_MAX && position >= 0 && position < n) {
                sel[position] = static_cast<int32_t>(i);
            }
        }
    }
}
