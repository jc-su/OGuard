#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ part_sums,
    const double* __restrict__ total_sum,
    double threshold_fraction,
    int64_t part_sums_len,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_values,
    int64_t* __restrict__ out_pos
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= part_sums_len) {
        return;
    }

    double v = part_sums[idx];
    double threshold = total_sum[0] * threshold_fraction;
    if (v > threshold) {
        unsigned long long pos = atomicAdd(
            reinterpret_cast<unsigned long long*>(out_pos),
            static_cast<unsigned long long>(1)
        );
        out_keys[static_cast<int64_t>(pos)] = idx;
        out_values[static_cast<int64_t>(pos)] = v;
    }
}
