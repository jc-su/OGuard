#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    uint32_t* __restrict__ bitmap,
    int64_t n_elements,
    int32_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t g = group_ids[idx];
        int64_t s = supp_ids[idx];
        
        int32_t word_idx = s / 32;
        int32_t bit_idx = s % 32;
        
        // Coalesced access layout: bitmap[word_idx * num_groups + g]
        int64_t offset = (int64_t)word_idx * num_groups + g;
        atomicOr(&bitmap[offset], 1u << bit_idx);
    }
}
