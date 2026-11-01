// q6_sum: the aggregate of TPC-H Q6.
//
// result[0] += sum of price[i] * discount[i] over the rows i with
// date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi, and
// quantity[i] < qty_max. Each block adds its partial sum to result[0] with one
// atomicAdd, so result[0] must be zero on entry. The kernel requires
// blockDim.x == BLOCK.

#include <cstdint>

constexpr int BLOCK = 256;
constexpr int WARPS = BLOCK / 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

// Returns the sum of v over the 32 lanes of the warp in lane 0.
__device__ __forceinline__ double warp_sum(double v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

extern "C" __global__ void __launch_bounds__(BLOCK)
kernel(const int32_t* __restrict__ shipdate, const double* __restrict__ discount,
       const double* __restrict__ quantity, const double* __restrict__ price,
       int32_t date_lo, int32_t date_hi, double disc_lo, double disc_hi, double qty_max,
       double* __restrict__ result, long long n)
{
    __shared__ double warp_sums[WARPS];

    double acc = 0.0;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const int32_t day = shipdate[i];
        const double disc = discount[i];
        if (date_lo <= day && day < date_hi && disc_lo <= disc && disc <= disc_hi &&
            quantity[i] < qty_max) {
            acc += price[i] * disc;
        }
    }

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    acc = warp_sum(acc);
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();

    // Warp 0 sums the warp sums and adds the block sum to the result.
    if (warp == 0) {
        acc = warp_sum(lane < WARPS ? warp_sums[lane] : 0.0);
        if (lane == 0) {
            atomicAdd(result, acc);
        }
    }
}
