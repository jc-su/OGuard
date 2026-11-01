#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* lengths,
    int32_t* offsets,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= n) {
        if (idx == 0) {
            offsets[0] = 0;
        } else {
            int64_t total = 0;
            for (int64_t i = 0; i < idx; i++) {
                total += lengths[i];
            }
            offsets[idx] = total;
        }
    }
}
