#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ supp_keys,
    const int32_t* __restrict__ supp_ids,
    int64_t n,
    int64_t max_suppkey,
    int32_t* __restrict__ supp_map
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int64_t k = supp_keys[i];
    if (k >= 0 && k <= max_suppkey) {
        supp_map[k] = supp_ids[i];
    }
}
