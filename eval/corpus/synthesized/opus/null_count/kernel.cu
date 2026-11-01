// Null count: add the number of rows whose validity bit is 0 to count[0].
#include <cstdint>

extern "C" __global__ void kernel(const uint8_t* valid, int64_t* count, int64_t n) {
    const int64_t nbytes = (n + 7) / 8;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t nulls = 0;
    for (int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < nbytes;
         j += stride) {
        const int rows = n - j * 8 < 8 ? static_cast<int>(n - j * 8) : 8;
        nulls += rows - __popc(valid[j] & ((1u << rows) - 1u));
    }

    // Sum over the warp, then over the warps of the block, then one atomic.
    for (int offset = 16; offset > 0; offset /= 2) {
        nulls += __shfl_down_sync(0xFFFFFFFFu, nulls, offset);
    }
    __shared__ int64_t partial[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x / 32u;
    if (lane == 0) partial[warp] = nulls;
    __syncthreads();
    if (warp == 0) {
        nulls = lane < blockDim.x / 32u ? partial[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            nulls += __shfl_down_sync(0xFFFFFFFFu, nulls, offset);
        }
        if (lane == 0 && nulls != 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(count),
                      static_cast<unsigned long long>(nulls));
        }
    }
}
