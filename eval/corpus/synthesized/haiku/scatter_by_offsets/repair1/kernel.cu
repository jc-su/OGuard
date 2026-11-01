#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int8_t* flags, const int32_t* offsets, int64_t n, int64_t total,
    int32_t* out)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (flags[i] != 0) {
            int32_t offset = offsets[i];
            if (offset >= 0 && offset < total) {
                out[offset] = (int32_t)i;
            }
        }
    }
}
