#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const uint32_t* keys, int64_t n, int64_t shift,
    int32_t* counts)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t key = keys[i];
        uint32_t byte_val = (key >> shift) & 255;
        atomicAdd(counts + byte_val, 1);
    }
}
