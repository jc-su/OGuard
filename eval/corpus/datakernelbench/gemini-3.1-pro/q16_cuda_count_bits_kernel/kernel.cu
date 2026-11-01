#include <cstdint>

extern "C" __global__ void kernel(
    const uint32_t* __restrict__ bitmap,
    int64_t* __restrict__ counts,
    int32_t num_groups,
    int32_t num_words
) {
    int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < num_groups) {
        int64_t count = 0;
        for (int32_t w = 0; w < num_words; ++w) {
            int64_t offset = (int64_t)w * num_groups + g;
            count += __popc(bitmap[offset]);
        }
        counts[g] = count;
    }
}
