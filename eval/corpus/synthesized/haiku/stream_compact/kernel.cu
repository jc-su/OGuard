#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* values,
    const int8_t* keep,
    int64_t* out,
    int32_t* count,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (keep[idx] != 0) {
            int32_t pos = atomicAdd(count, 1);
            out[pos] = values[idx];
        }
    }
}
