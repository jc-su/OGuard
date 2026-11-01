// tile_scan: exclusive prefix sum within tiles of 1024 consecutive rows.
// out[i] is the sum of values[j] over the rows j < i of row i's tile, and
// totals[t] is the sum of tile t. One block of 256 threads scans one tile, so
// the grid has (n + 1023) / 1024 blocks. Sums wrap around in 32-bit two's
// complement arithmetic.
#include <cstdint>

constexpr int kTile = 1024;
constexpr int kThreads = 256;
constexpr int kItems = kTile / kThreads;  // Consecutive rows per thread.
constexpr int kWarps = kThreads / 32;

// Shared memory index of tile row e. One pad word after every 32 words puts
// the rows 4t + j that the 32 threads of a warp read at once in distinct banks.
__device__ __forceinline__ int padded(int e)
{
    return e + (e >> 5);
}

extern "C" __global__ void __launch_bounds__(kThreads)
kernel(const int32_t* __restrict__ values,
       int32_t* __restrict__ out,
       int32_t* __restrict__ totals,
       int64_t n)
{
    __shared__ uint32_t tile[kTile + kTile / 32];
    __shared__ uint32_t warp_scan[kWarps];

    const int64_t base = static_cast<int64_t>(blockIdx.x) * kTile;
    const int count = n - base < kTile ? static_cast<int>(n - base) : kTile;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    // Coalesced load of the tile. Rows past the end of the input count as 0.
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        const int e = t + j * kThreads;
        tile[padded(e)] = e < count ? static_cast<uint32_t>(values[base + e]) : 0u;
    }
    __syncthreads();

    // Thread t owns the tile rows 4t .. 4t + 3.
    uint32_t item[kItems];
    uint32_t thread_sum = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        item[j] = tile[padded(kItems * t + j)];
        thread_sum += item[j];
    }

    // Inclusive scan of the thread sums within each warp.
    uint32_t inclusive = thread_sum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inclusive, d);
        if (lane >= d) inclusive += up;
    }
    if (lane == 31) warp_scan[warp] = inclusive;
    __syncthreads();

    // The first warp replaces the warp sums by their inclusive scan.
    if (warp == 0) {
        uint32_t sum = lane < kWarps ? warp_scan[lane] : 0u;
#pragma unroll
        for (int d = 1; d < kWarps; d <<= 1) {
            const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, sum, d);
            if (lane >= d) sum += up;
        }
        if (lane < kWarps) warp_scan[lane] = sum;
    }
    __syncthreads();

    // Exclusive prefix of each of the thread's rows, written back in place.
    uint32_t running = (warp > 0 ? warp_scan[warp - 1] : 0u) + inclusive - thread_sum;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        tile[padded(kItems * t + j)] = running;
        running += item[j];
    }
    if (t == 0) totals[blockIdx.x] = static_cast<int32_t>(warp_scan[kWarps - 1]);
    __syncthreads();

    // Coalesced store of the rows that exist.
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        const int e = t + j * kThreads;
        if (e < count) out[base + e] = static_cast<int32_t>(tile[padded(e)]);
    }
}
