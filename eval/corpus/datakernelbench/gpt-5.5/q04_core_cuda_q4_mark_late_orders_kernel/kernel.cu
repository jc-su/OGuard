#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commitdate_days,
    const int32_t* __restrict__ l_receiptdate_days,
    uint8_t* __restrict__ has_late_lineitem,
    int64_t n_lineitem,
    int64_t max_key
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        int32_t commit_day = l_commitdate_days[idx];
        int32_t receipt_day = l_receiptdate_days[idx];
        if (commit_day < receipt_day) {
            int64_t key = l_orderkey[idx];
            if (key >= 0 && key <= max_key) {
                has_late_lineitem[key] = static_cast<uint8_t>(1);
            }
        }
    }
}
