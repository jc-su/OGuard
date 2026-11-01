// q6_sum: the aggregate of TPC-H Q6.
//
// result[0] += sum of price[i] * discount[i] over the rows i with
// date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi, and
// quantity[i] < qty_max. Each warp adds its partial sum to a shared block sum
// with one atomicAdd, and each block adds its block sum to result[0] with one
// atomicAdd, so result[0] must be zero on entry. The kernel requires a block
// size that is a multiple of 32 and at most BLOCK.

#include <cstdint>

constexpr int BLOCK = 256;  // Largest number of threads per block.
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
    __shared__ double block_sum;
    if (threadIdx.x == 0) {
        block_sum = 0.0;
    }

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

    acc = warp_sum(acc);
    // Thread 0 has set block_sum to zero before any warp adds to it.
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&block_sum, acc);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}
