#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_nationkey,
    const int64_t* __restrict__ nation_keys,
    int64_t n_suppliers,
    int64_t n_nation_keys,
    uint8_t* __restrict__ supplier_keep,
    int64_t supplier_keep_len
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_suppliers) {
        return;
    }

    int64_t nk = s_nationkey[idx];
    bool keep = false;
    #pragma unroll 1
    for (int64_t j = 0; j < n_nation_keys; ++j) {
        if (nk == nation_keys[j]) {
            keep = true;
            break;
        }
    }

    if (keep) {
        int64_t sk = s_suppkey[idx];
        if (sk >= 0 && sk < supplier_keep_len) {
            supplier_keep[sk] = static_cast<uint8_t>(1);
        }
    }
}
