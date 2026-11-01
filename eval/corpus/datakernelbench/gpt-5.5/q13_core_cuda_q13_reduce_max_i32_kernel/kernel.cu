#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int32_t* __restrict__ inp,
    int64_t n,
    int32_t* __restrict__ out
) {
    __shared__ int32_t smem[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int32_t v = 0;
    if (idx < n) {
        v = inp[idx];
    }
    smem[threadIdx.x] = v;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int32_t other = smem[threadIdx.x + stride];
            if (other > smem[threadIdx.x]) {
                smem[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = smem[0];
    }
}
