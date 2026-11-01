#include <cstdint>

extern "C" __global__ void kernel(int64_t* __restrict__ out, int64_t limit) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < limit) {
        out[idx] = idx;
    }
}
