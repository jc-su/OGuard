#include <cstdint>

extern "C" __global__ void kernel(const uint64_t* __restrict__ ps_key,
                               const int64_t* __restrict__ lower_idx,
                               const uint64_t* __restrict__ uniq_key,
                               const double* __restrict__ uniq_sum,
                               const int64_t* __restrict__ ps_availqty,
                               uint8_t* __restrict__ mask,
                               int64_t n_ps,
                               int64_t n_groups) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_ps) {
        uint64_t k = ps_key[i];
        int64_t pos = lower_idx[i];
        double sum = 0.0;
        if (pos < n_groups && uniq_key[pos] == k) {
            sum = uniq_sum[pos];
        }
        bool keep = (sum > 0.0) && (static_cast<double>(ps_availqty[i]) > 0.5 * sum);
        mask[i] = keep ? 1 : 0;
    }
}
