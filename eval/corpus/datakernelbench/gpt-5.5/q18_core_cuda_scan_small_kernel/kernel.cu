#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ in,
    int64_t* __restrict__ out,
    int64_t n
) {
    __shared__ int64_t temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t x = (tid < n) ? in[tid] : 0;
    temp[tid] = x;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t add = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += add;
        __syncthreads();
    }

    if (tid < n) {
        out[tid] = temp[tid] - x;
    }
}
