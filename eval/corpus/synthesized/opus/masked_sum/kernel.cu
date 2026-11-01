#include <cstdint>

// Grid-stride accumulation per thread, a warp-shuffle and shared-memory
// reduction per block, and one atomic add per block. Launch with 256 threads
// per block and any grid size of at least one block.
extern "C" __global__ void __launch_bounds__(256)
kernel(const double* __restrict__ values, const int8_t* __restrict__ mask, double* result,
       int64_t n)
{
    constexpr int WARPS = 256 / 32;
    __shared__ double warp_sums[WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double acc = 0.0;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        if (mask[i] != 0) {
            acc += values[i];
        }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, delta);
    }
    if (lane == 0) {
        warp_sums[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        acc = lane < WARPS ? warp_sums[lane] : 0.0;
#pragma unroll
        for (int delta = WARPS / 2; delta > 0; delta >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, delta);
        }
        if (lane == 0) {
            atomicAdd(result, acc);
        }
    }
}
