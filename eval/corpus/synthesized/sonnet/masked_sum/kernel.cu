#include <cstdint>

// Each thread sums its rows, each block reduces its threads with warp shuffles and shared memory,
// and one thread per block adds the block sum to result[0].
extern "C" __global__ void kernel(const double* values, const int8_t* mask, double* result,
                                  int64_t n) {
    __shared__ double warp_sums[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    double s = 0.0;
    for (int64_t i = start; i < n; i += stride)
        if (mask[i] != 0) s += values[i];
    for (int d = 16; d > 0; d >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, d);
    if (lane == 0) warp_sums[wid] = s;
    __syncthreads();
    if (wid == 0) {
        s = lane < (int)(blockDim.x >> 5) ? warp_sums[lane] : 0.0;
        for (int d = 16; d > 0; d >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, d);
        if (lane == 0) atomicAdd(result, s);
    }
}
