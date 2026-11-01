// filter_select: write the row ids i with quantity[i] < threshold into sel.
//
// The kernel processes the rows in tiles of TILE rows. Each block reserves one
// range of sel per tile with a single atomicAdd on count, so the kernel
// requires count[0] == 0 on entry. The caller sets count[0] to zero before the
// launch (see run_filter_select in test.cu). The kernel requires
// blockDim.x == BLOCK.

#include <cstdint>

constexpr int BLOCK = 256;
constexpr int WARPS = BLOCK / 32;
constexpr int ITEMS = 4;  // Rows per thread in one tile.
constexpr long long TILE = BLOCK * ITEMS;
constexpr unsigned FULL_MASK = 0xffffffffu;

extern "C" __global__ void __launch_bounds__(BLOCK)
kernel(const double* __restrict__ quantity, double threshold,
       int32_t* __restrict__ sel, int32_t* __restrict__ count, long long n)
{
    __shared__ int warp_matches[WARPS];
    __shared__ int warp_slot[WARPS];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const unsigned lanes_below = (1u << lane) - 1u;

    // The loop bounds are the same for all threads of a block, so every
    // thread reaches each __syncthreads call.
    for (long long tile = blockIdx.x * TILE; tile < n; tile += gridDim.x * TILE) {
        // Evaluate the predicate and count the matches of each warp.
        unsigned masks[ITEMS];
        int matches = 0;
#pragma unroll
        for (int k = 0; k < ITEMS; ++k) {
            const long long i = tile + k * BLOCK + tid;
            const bool keep = i < n && quantity[i] < threshold;
            masks[k] = __ballot_sync(FULL_MASK, keep);
            matches += __popc(masks[k]);
        }
        if (lane == 0) {
            warp_matches[warp] = matches;
        }
        __syncthreads();

        // Reserve one range of sel for the tile and split it among the warps.
        if (tid == 0) {
            int total = 0;
            for (int w = 0; w < WARPS; ++w) {
                total += warp_matches[w];
            }
            int slot = atomicAdd(count, total);
            for (int w = 0; w < WARPS; ++w) {
                warp_slot[w] = slot;
                slot += warp_matches[w];
            }
        }
        __syncthreads();

        // Each matching lane writes its row id after the matches of the lower
        // lanes of its warp.
        int slot = warp_slot[warp];
#pragma unroll
        for (int k = 0; k < ITEMS; ++k) {
            if ((masks[k] >> lane) & 1u) {
                sel[slot + __popc(masks[k] & lanes_below)] =
                    static_cast<int32_t>(tile + k * BLOCK + tid);
            }
            slot += __popc(masks[k]);
        }
        // Do not overwrite the shared arrays before all warps read them.
        __syncthreads();
    }
}
