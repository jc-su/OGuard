#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(uint32_t *keys, int32_t *cursors, int32_t *out, int64_t n, int64_t shift) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t key = keys[idx];
        uint32_t byte_val = (key >> shift) & 0xFF;
        int32_t pos = atomicAdd(&cursors[byte_val], 1);
        out[pos] = (int32_t)idx;
    }
}
