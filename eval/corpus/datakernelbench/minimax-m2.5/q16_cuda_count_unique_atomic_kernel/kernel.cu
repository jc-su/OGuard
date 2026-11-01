#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ sorted_packed_pairs,
    int64_t* __restrict__ group_counts,
    int64_t num_pairs
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    int64_t packed = sorted_packed_pairs[idx];
    int64_t group_id = packed >> 32;
    int64_t supp_id = packed & 0xFFFFFFFFLL;
    
    // Check if this is a new unique pair (different from previous)
    bool is_new_unique = true;
    if (idx > 0) {
        int64_t prev_packed = sorted_packed_pairs[idx - 1];
        int64_t prev_group = prev_packed >> 32;
        int64_t prev_supp = prev_packed & 0xFFFFFFFFLL;
        if (group_id == prev_group && supp_id == prev_supp) {
            is_new_unique = false;
        }
    }
    
    if (is_new_unique) {
        // Use atomicAdd with unsigned long long cast for int64
        atomicAdd((unsigned long long*)&group_counts[group_id], 1ULL);
    }
}
