#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    int32_t* __restrict__ counts,
    int64_t num_customers
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_custkey[idx];
        if (key >= 0 && key <= num_customers) {
            atomicAdd(counts + key, 1);
        }
    }
}
