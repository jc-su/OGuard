#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    uint8_t* __restrict__ flags,
    int64_t n_orders,
    int64_t flags_len
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key < flags_len) {
            flags[key] = static_cast<uint8_t>(1);
        }
    }
}
