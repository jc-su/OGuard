#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ ps_sum_qty,
    uint8_t* __restrict__ supp_mask,
    int64_t nps,
    int64_t max_suppkey
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nps) {
        return;
    }

    double sum_qty = ps_sum_qty[idx];
    if (sum_qty > 0.0 && ((double)ps_availqty[idx]) > 0.5 * sum_qty) {
        int64_t supp = ps_suppkey[idx];
        if (supp >= 0 && supp <= max_suppkey) {
            supp_mask[supp] = (uint8_t)1;
        }
    }
}
