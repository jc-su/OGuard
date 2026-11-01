#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ sums,
    int64_t* __restrict__ out_all_keys,
    int64_t* __restrict__ out_big_keys,
    double* __restrict__ out_big_sums,
    int32_t* __restrict__ count_all,
    int32_t* __restrict__ count_big,
    int64_t num_slots,
    double threshold
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_slots) {
        double s = sums[idx];
        if (s > 0.0) {
            int32_t pos = atomicAdd(count_all, 1);
            out_all_keys[pos] = idx;
        }
        if (s > threshold) {
            int32_t pos = atomicAdd(count_big, 1);
            out_big_keys[pos] = idx;
            out_big_sums[pos] = s;
        }
    }
}
