#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    const uint8_t* __restrict__ supp_mask,
    int64_t* __restrict__ out_row_idx,
    int64_t* __restrict__ out_count,
    int64_t ns,
    int64_t max_suppkey
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ns) {
        return;
    }

    int64_t supp = s_suppkey[idx];
    if (supp >= 0 && supp <= max_suppkey && supp_mask[supp] != 0) {
        unsigned long long pos = atomicAdd((unsigned long long*)out_count, 1ULL);
        out_row_idx[(int64_t)pos] = idx;
    }
}
