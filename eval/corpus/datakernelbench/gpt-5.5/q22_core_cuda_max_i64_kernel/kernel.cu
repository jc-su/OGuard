#include <climits>
#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ data,
    int64_t* __restrict__ block_max,
    int64_t n
) {
    __shared__ int64_t smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t v = LLONG_MIN;
    if (idx < n) {
        v = data[idx];
    }
    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int64_t other = smem[tid + stride];
            if (other > smem[tid]) {
                smem[tid] = other;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}
