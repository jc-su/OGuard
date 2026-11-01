#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    uint8_t* __restrict__ has_order,
    int64_t n_orders,
    int64_t key_space
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key < key_space) {
            has_order[key] = 1;
        }
    }
}
