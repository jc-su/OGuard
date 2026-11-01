// Prefix offsets: offsets[0] = 0 and offsets[i + 1] = offsets[i] + lengths[i].
#include <cstdint>

constexpr int kBlockThreads = 256;
constexpr int kWarps = kBlockThreads / 32;
constexpr int kItems = 16;
constexpr int kTile = kBlockThreads * kItems;
// One padding word per 32 words keeps the per-thread reads free of bank conflicts.
constexpr int kPaddedTile = kTile + kTile / 32;

__device__ __forceinline__ int padded(int j) { return j + (j >> 5); }

// The signature has no scratch buffer to pass a carry between blocks, so block 0
// scans the whole column tile by tile and the other blocks have no work. The
// kernel expects blockDim.x == kBlockThreads. The sums are unsigned so that they
// wrap modulo 2^32 like the int32 result.
extern "C" __global__ void __launch_bounds__(kBlockThreads)
    kernel(const int32_t* lengths, int32_t* offsets, int64_t n) {
    if (blockIdx.x != 0) return;
    __shared__ uint32_t buf[kPaddedTile];
    __shared__ uint32_t warp_sums[kWarps];
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    if (t == 0) offsets[0] = 0;

    // Coalesced load of the first tile.
    uint32_t loaded[kItems];
#pragma unroll
    for (int q = 0; q < kItems; ++q) {
        const int64_t row = t + q * kBlockThreads;
        loaded[q] = row < n ? static_cast<uint32_t>(lengths[row]) : 0u;
    }
    uint32_t carry = 0;
    for (int64_t base = 0; base < n; base += kTile) {
#pragma unroll
        for (int q = 0; q < kItems; ++q) buf[padded(t + q * kBlockThreads)] = loaded[q];
        __syncthreads();

        // Issue the loads of the next tile; they complete while this tile is scanned.
#pragma unroll
        for (int q = 0; q < kItems; ++q) {
            const int64_t row = base + kTile + t + q * kBlockThreads;
            loaded[q] = row < n ? static_cast<uint32_t>(lengths[row]) : 0u;
        }

        // Each thread scans its kItems consecutive elements in registers.
        uint32_t scanned[kItems];
        uint32_t total = 0;
#pragma unroll
        for (int q = 0; q < kItems; ++q) {
            total += buf[padded(t * kItems + q)];
            scanned[q] = total;
        }

        // Inclusive scan of the thread totals over the warp, then over the warps.
        uint32_t inclusive = total;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inclusive, d);
            if (lane >= d) inclusive += up;
        }
        if (lane == 31) warp_sums[warp] = inclusive;
        __syncthreads();
        if (warp == 0) {
            uint32_t warp_total = lane < kWarps ? warp_sums[lane] : 0u;
#pragma unroll
            for (int d = 1; d < kWarps; d *= 2) {
                const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, warp_total, d);
                if (lane >= d) warp_total += up;
            }
            if (lane < kWarps) warp_sums[lane] = warp_total;
        }
        __syncthreads();

        const uint32_t prefix =
            carry + (inclusive - total) + (warp > 0 ? warp_sums[warp - 1] : 0u);
#pragma unroll
        for (int q = 0; q < kItems; ++q) buf[padded(t * kItems + q)] = scanned[q] + prefix;
        carry += warp_sums[kWarps - 1];
        __syncthreads();

        // Coalesced store of the tile.
#pragma unroll
        for (int q = 0; q < kItems; ++q) {
            const int j = t + q * kBlockThreads;
            if (base + j < n) offsets[base + j + 1] = static_cast<int32_t>(buf[padded(j)]);
        }
        // buf and warp_sums are written again by the next tile.
        __syncthreads();
    }
}
