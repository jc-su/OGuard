#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(uint8_t *valid, int32_t *rows, int32_t *count, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t byte_idx = idx / 8;
        int64_t bit_idx = idx % 8;
        uint8_t byte_val = valid[byte_idx];
        if ((byte_val >> bit_idx) & 1) {
            int32_t pos = atomicAdd(count, 1);
            rows[pos] = (int32_t)idx;
        }
    }
}
