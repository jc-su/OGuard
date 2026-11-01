#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int32_t *codes, double *dictionary, double *out, int64_t n, int64_t d) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t code = codes[idx];
        out[idx] = dictionary[code];
    }
}
