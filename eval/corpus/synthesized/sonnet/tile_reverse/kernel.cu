#include <cstdint>

#define TILE 1024

// Row i of tile t takes the value at the mirrored position of the same tile. Consecutive threads
// read consecutive addresses in reverse order, which coalesces as well as a forward copy.
extern "C" __global__ void kernel(const int32_t* values, int32_t* out, int64_t n) {
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start; i < n; i += stride) {
        const int64_t base = i - i % TILE;
        const int64_t rest = n - base;
        const int64_t length = rest < TILE ? rest : TILE;
        out[i] = values[base + length - 1 - (i - base)];
    }
}
