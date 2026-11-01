#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ part_keys,
    const bool* __restrict__ promo_flags,
    uint8_t* __restrict__ lookup,
    int64_t min_key,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t off = part_keys[idx] - min_key;
        lookup[off] = promo_flags[idx] ? static_cast<uint8_t>(2) : static_cast<uint8_t>(1);
    }
}
