// block_sort: sort the keys of each tile of 1024 consecutive rows ascending
// and write them to the same rows of out. One block of 256 threads sorts one
// tile with a bitonic sorting network, so the grid has (n + 1023) / 1024
// blocks.
//
// Thread t holds the tile rows 4t .. 4t + 3 in registers. Stage (size, stride)
// of the network compares rows e and e + stride for each e with bit `stride`
// clear, and orders the pair ascending when bit `size` of e is clear. A stage
// runs where the two rows of each pair are:
//   stride 1 and 2:    in the registers of one thread;
//   stride 4 to 64:    in threads of one warp, exchanged by warp shuffles;
//   stride 128 to 512: in different warps, compared in shared memory.
//
// The rows past the end of a short last tile are filled with INT32_MAX, which
// sorts them after the real keys; their values equal any INT32_MAX key, so the
// first count rows of the sorted tile are the sorted keys.
#include <cstdint>

constexpr int kTile = 1024;
constexpr int kThreads = 256;
constexpr int kItems = kTile / kThreads;      // Rows per thread.
constexpr int kPairs = kTile / 2 / kThreads;  // Shared memory pairs per thread per stage.
constexpr int kBlockStride = kItems * 32;     // Smallest stride that crosses warps.

// Order x and y ascending or descending.
__device__ __forceinline__ void order(int32_t& x, int32_t& y, bool ascending)
{
    const int32_t small = min(x, y);
    const int32_t large = max(x, y);
    x = ascending ? small : large;
    y = ascending ? large : small;
}

extern "C" __global__ void __launch_bounds__(kThreads)
kernel(const int32_t* __restrict__ keys, int32_t* __restrict__ out, int64_t n)
{
    __shared__ __align__(16) int32_t tile[kTile];
    int4* const rows = reinterpret_cast<int4*>(tile);  // rows[t] holds rows 4t .. 4t + 3.

    const int64_t base = static_cast<int64_t>(blockIdx.x) * kTile;
    const int count = n - base < kTile ? static_cast<int>(n - base) : kTile;
    const int t = threadIdx.x;

    // Coalesced load of the tile, then each thread takes its four rows.
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        const int e = t + j * kThreads;
        tile[e] = e < count ? keys[base + e] : INT32_MAX;
    }
    __syncthreads();
    int4 own = rows[t];
    int32_t v0 = own.x, v1 = own.y, v2 = own.z, v3 = own.w;

#pragma unroll
    for (int size = 2; size <= kTile; size <<= 1) {
        int stride = size >> 1;

        if (stride >= kBlockStride) {
            __syncthreads();  // Every thread has read its rows back from the last use of tile.
            rows[t] = make_int4(v0, v1, v2, v3);
            __syncthreads();
            for (; stride >= kBlockStride; stride >>= 1) {
#pragma unroll
                for (int q = 0; q < kPairs; ++q) {
                    const int p = t + q * kThreads;
                    const int lo = 2 * p - (p & (stride - 1));
                    int32_t a = tile[lo];
                    int32_t b = tile[lo + stride];
                    order(a, b, (lo & size) == 0);
                    tile[lo] = a;
                    tile[lo + stride] = b;
                }
                __syncthreads();
            }
            own = rows[t];
            v0 = own.x;
            v1 = own.y;
            v2 = own.z;
            v3 = own.w;
        }

        // For stride >= 2 all four rows of the thread share bit `size`.
        const bool ascending = ((kItems * t) & size) == 0;
#pragma unroll
        for (; stride >= kItems; stride >>= 1) {
            // The partner of row 4t + r is row 4(t ^ lane_mask) + r.
            const int lane_mask = stride / kItems;
            const bool keep_min = ((t & lane_mask) == 0) == ascending;
            const int32_t p0 = __shfl_xor_sync(0xFFFFFFFFu, v0, lane_mask);
            const int32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, v1, lane_mask);
            const int32_t p2 = __shfl_xor_sync(0xFFFFFFFFu, v2, lane_mask);
            const int32_t p3 = __shfl_xor_sync(0xFFFFFFFFu, v3, lane_mask);
            v0 = keep_min ? min(v0, p0) : max(v0, p0);
            v1 = keep_min ? min(v1, p1) : max(v1, p1);
            v2 = keep_min ? min(v2, p2) : max(v2, p2);
            v3 = keep_min ? min(v3, p3) : max(v3, p3);
        }
        if (stride == 2) {
            order(v0, v2, ascending);
            order(v1, v3, ascending);
        }
        order(v0, v1, ascending);
        order(v2, v3, ((kItems * t + 2) & size) == 0);
    }

    // Each thread puts its rows back, then the block stores the tile coalesced.
    __syncthreads();
    rows[t] = make_int4(v0, v1, v2, v3);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        const int e = t + j * kThreads;
        if (e < count) out[base + e] = tile[e];
    }
}
